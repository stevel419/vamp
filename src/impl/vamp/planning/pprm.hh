#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <utility>
#include <vector>

#include <vamp/collision/environment.hh>
#include <vamp/planning/nn.hh>
#include <vamp/planning/plan.hh>
#include <vamp/planning/roadmap.hh>
#include <vamp/planning/validate.hh>
#include <vamp/random/rng.hh>
#include <vamp/utils.hh>
#include <vamp/vector.hh>

// Without OpenMP (VAMP_OPENMP=OFF) every loop below runs serially and the results are unchanged.
#if defined(_OPENMP)
#define VAMP_OMP(directive) _Pragma(#directive)
#else
#define VAMP_OMP(directive)
#endif

namespace vamp::planning
{
    // Where PPRM::build_roadmap() spent its time, summed over all batches.
    struct PPRMStats
    {
        std::size_t sample_nanoseconds{0};
        std::size_t neighbor_nanoseconds{0};
        std::size_t validate_nanoseconds{0};
        std::size_t insert_nanoseconds{0};
        std::size_t edges_checked{0};
        std::size_t batches{0};
    };

    // Flat form of a roadmap's connectivity: every undirected edge exactly once, plus a CSR adjacency
    // that remembers which edge each neighbor entry came from.
    struct RoadmapGraph
    {
        using Pair = std::array<std::uint32_t, 2>;

        std::vector<Pair> edges;             // {i, j} with i < j
        std::vector<std::uint32_t> offsets;  // adjacency of vertex v is [offsets[v], offsets[v + 1])
        std::vector<Pair> adjacency;         // {neighbor vertex, edge index}
    };

    // One entry per vertex and per edge of a RoadmapGraph; non-zero means collision-free.
    struct RoadmapValidity
    {
        std::vector<char> vertices;
        std::vector<char> edges;
    };

    // Batched PRM. build_roadmap() produces the same roadmap as PRM::build_roadmap() for the same
    // sampler state, for any batch size (settings.batch_size) and any number of threads; only the
    // collision checking is spread over threads.
    template <
        typename Robot,
        std::size_t rake,
        std::size_t resolution,
        typename NeighborParamsT = PRMStarNeighborParams>
    struct PPRM
    {
        using Configuration = typename Robot::Configuration;
        using Environment = collision::Environment<FloatVector<rake>>;
        static constexpr auto dimension = Robot::dimension;
        using RNG = typename vamp::rng::RNG<Robot>;

        // Check edge(e) -> {from, to} for every e in [0, n), in parallel. Each edge is checked exactly
        // as the serial planner checks it, so the answers do not depend on the thread count.
        template <typename EdgeFn>
        inline static void validate_edges(
            std::size_t n,
            const EdgeFn &edge,
            const Environment &environment,
            std::vector<char> &valid) noexcept
        {
            valid.resize(n);
            // fkcc_attach() poses the attachment through a mutable member of the shared environment
            const bool parallel = not environment.attachments;
            VAMP_OMP(omp parallel for schedule(runtime) if (parallel))
            for (std::size_t e = 0; e < n; ++e)
            {
                const std::pair<Configuration, Configuration> endpoints = edge(e);
                valid[e] =
                    validate_motion<Robot, rake, resolution>(endpoints.first, endpoints.second, environment);
            }
        }

        inline static auto build_roadmap(
            const Configuration &start,
            const Configuration &goal,
            const Environment &environment,
            const RoadmapSettings<NeighborParamsT> &settings,
            typename RNG::Ptr rng,
            PPRMStats *stats = nullptr) noexcept -> Roadmap<Robot>
        {
            using Clock = std::chrono::steady_clock;
            using Candidates = std::vector<std::pair<NNNode<dimension>, float>>;

            NN<dimension> roadmap;

            constexpr const unsigned int start_index = 0;
            constexpr const unsigned int goal_index = 1;

            auto start_time = Clock::now();

            auto states = std::unique_ptr<float, decltype(&free)>(
                vamp::utils::vector_alloc<float, FloatVectorAlignment, FloatVectorWidth>(
                    settings.max_samples * Configuration::num_scalars_rounded),
                &free);
            std::vector<RoadmapNode> nodes;
            nodes.reserve(settings.max_samples);

            const auto state_index = [&states](std::size_t index) -> float *
            { return states.get() + index * Configuration::num_scalars_rounded; };

            // Add start and goal to structures
            start.to_array(state_index(start_index));
            goal.to_array(state_index(goal_index));
            nodes.emplace_back(start_index, start_index, 0.0);
            nodes.emplace_back(goal_index, goal_index);
            roadmap.insert(NNNode<dimension>{start_index, {state_index(start_index)}});
            roadmap.insert(NNNode<dimension>{goal_index, {state_index(goal_index)}});

            const std::size_t batch_size = std::max<std::size_t>(settings.batch_size, 1);
            std::vector<Candidates> candidates(batch_size);
            std::vector<std::size_t> offsets(batch_size + 1, 0);
            std::vector<std::array<std::uint32_t, 2>> flat;  // {batch node, candidate}
            std::vector<char> valid;

            PPRMStats local;
            std::size_t draws = 0;
            const auto lap = [](Clock::time_point &mark) -> std::size_t
            {
                const auto now = Clock::now();
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - mark).count();
                mark = now;
                return elapsed;
            };

            while (draws < settings.max_iterations and nodes.size() < settings.max_samples)
            {
                auto mark = Clock::now();
                const std::size_t n0 = nodes.size();

                // 1. Sample: keep the next valid samples, in generator order. A round never draws more
                // samples than the serial planner would, so the sampler ends in the same state.
                std::size_t b = 0;
                while (b < batch_size)
                {
                    const std::size_t m = std::min(
                        {batch_size - b, settings.max_samples - (n0 + b), settings.max_iterations - draws});
                    if (m == 0)
                    {
                        break;
                    }

                    for (std::size_t c = 0; c < m; ++c)
                    {
                        rng->next().to_array(state_index(n0 + b + c));
                    }

                    draws += m;
                    valid.resize(m);
                    VAMP_OMP(omp parallel for schedule(static) if (m >= 64))
                    for (std::size_t c = 0; c < m; ++c)
                    {
                        const Configuration temp(state_index(n0 + b + c));
                        typename Robot::template ConfigurationBlock<rake> temp_block;
                        for (auto i = 0U; i < dimension; ++i)
                        {
                            temp_block[i] = temp.broadcast(i);
                        }

                        valid[c] = Robot::template fkcc<rake>(environment, temp_block);
                    }

                    const std::size_t first = n0 + b;
                    for (std::size_t c = 0; c < m; ++c)
                    {
                        if (valid[c])
                        {
                            if (first + c != n0 + b)
                            {
                                std::copy_n(
                                    state_index(first + c),
                                    Configuration::num_scalars_rounded,
                                    state_index(n0 + b));
                            }

                            ++b;
                        }
                    }
                }

                local.sample_nanoseconds += lap(mark);

                // 2. Find candidates: node n0 + j sees the n0 nodes in the tree plus batch nodes
                // 0..j-1, with k and r for a roadmap of size n0 + j - what the serial planner sees.
                // The tree is only read here; nigh's nearest() is const and keeps its state local.
                VAMP_OMP(omp parallel for schedule(dynamic, 8) if (b >= 16))
                for (std::size_t j = 0; j < b; ++j)
                {
                    auto &found = candidates[j];
                    auto *state = state_index(n0 + j);
                    const auto k = settings.neighbor_params.max_neighbors(n0 + j);
                    const auto r = settings.neighbor_params.neighbor_radius(n0 + j);
                    roadmap.nearest(found, NNFloatArray<dimension>{state}, k, r);

                    const std::size_t from_tree = found.size();
                    const float bound = (from_tree > 0 and from_tree == k) ? found.back().second : r;
                    for (std::size_t i = 0; i < j; ++i)
                    {
                        auto *other = state_index(n0 + i);
                        const float distance = Space<dimension>::distance(
                            NNFloatArray<dimension>{other}, NNFloatArray<dimension>{state});
                        if (distance <= bound)
                        {
                            found.emplace_back(NNNode<dimension>{n0 + i, {other}}, distance);
                        }
                    }

                    if (found.size() > from_tree)
                    {
                        std::sort(
                            found.begin(),
                            found.end(),
                            [](const auto &a, const auto &b) { return a.second < b.second; });
                        if (found.size() > k)
                        {
                            found.resize(k);
                        }
                    }
                }

                flat.clear();
                for (std::size_t j = 0; j < b; ++j)
                {
                    offsets[j + 1] = offsets[j] + candidates[j].size();
                    for (std::size_t c = 0; c < candidates[j].size(); ++c)
                    {
                        flat.push_back({static_cast<std::uint32_t>(j), static_cast<std::uint32_t>(c)});
                    }
                }

                local.neighbor_nanoseconds += lap(mark);

                // 3. Validate every candidate edge of the batch
                validate_edges(
                    flat.size(),
                    [&](std::size_t e) -> std::pair<Configuration, Configuration>
                    {
                        const auto [j, c] = flat[e];
                        return {candidates[j][c].first.as_vector(), Configuration(state_index(n0 + j))};
                    },
                    environment,
                    valid);

                local.validate_nanoseconds += lap(mark);

                // 4. Insert, in the serial planner's order
                for (std::size_t j = 0; j < b; ++j)
                {
                    auto &node = nodes.emplace_back(nodes.size(), std::numeric_limits<unsigned int>::max());
                    for (std::size_t c = 0; c < candidates[j].size(); ++c)
                    {
                        if (valid[offsets[j] + c])
                        {
                            const auto &[neighbor, distance] = candidates[j][c];
                            node.neighbors.emplace_back(
                                typename RoadmapNode::Neighbor{
                                    static_cast<unsigned int>(neighbor.index), distance});
                            nodes[neighbor.index].neighbors.emplace_back(
                                typename RoadmapNode::Neighbor{node.index, distance});
                        }
                    }

                    roadmap.insert(NNNode<dimension>{node.index, {state_index(node.index)}});
                }

                local.insert_nanoseconds += lap(mark);
                local.edges_checked += flat.size();
                local.batches++;
            }

            Roadmap<Robot> result;
            result.vertices.reserve(nodes.size());

            for (const auto &node : nodes)
            {
                result.vertices.emplace_back(state_index(node.index));
                result.edges.emplace_back(std::vector<std::size_t>());

                for (const auto &nbr : node.neighbors)
                {
                    result.edges.back().emplace_back(nbr.index);
                }
            }

            result.nanoseconds = vamp::utils::get_elapsed_nanoseconds(start_time);
            // PRM::build_roadmap() counts one more iteration than it draws samples
            result.iterations = draws + 1;

            if (stats != nullptr)
            {
                *stats = local;
            }

            return result;
        }

        inline static auto graph(const Roadmap<Robot> &roadmap) -> RoadmapGraph
        {
            RoadmapGraph result;
            const std::size_t n = roadmap.vertices.size();
            result.offsets.assign(n + 1, 0);

            for (std::size_t j = 0; j < n; ++j)
            {
                for (const auto i : roadmap.edges[j])
                {
                    if (i < j)
                    {
                        result.edges.push_back(
                            {static_cast<std::uint32_t>(i), static_cast<std::uint32_t>(j)});
                        ++result.offsets[i + 1];
                        ++result.offsets[j + 1];
                    }
                }
            }

            std::partial_sum(result.offsets.begin(), result.offsets.end(), result.offsets.begin());
            result.adjacency.resize(2 * result.edges.size());
            auto cursor = result.offsets;
            for (std::uint32_t e = 0; e < result.edges.size(); ++e)
            {
                const auto [i, j] = result.edges[e];
                result.adjacency[cursor[i]++] = {j, e};
                result.adjacency[cursor[j]++] = {i, e};
            }

            return result;
        }

        // Recheck every vertex and every edge of a roadmap against an environment. An edge is valid
        // when both of its vertices are valid and the motion from the lower to the higher vertex
        // index is valid - the direction it was checked in when the roadmap was built.
        inline static auto
        revalidate(const Roadmap<Robot> &roadmap, const RoadmapGraph &graph, const Environment &environment)
            -> RoadmapValidity
        {
            RoadmapValidity validity;
            const std::size_t n = roadmap.vertices.size();
            const std::size_t n_edges = graph.edges.size();
            validity.vertices.resize(n);
            validity.edges.resize(n_edges);

            const bool parallel = not environment.attachments;
            VAMP_OMP(omp parallel for schedule(runtime) if (parallel))
            for (std::size_t v = 0; v < n; ++v)
            {
                validity.vertices[v] =
                    validate_motion<Robot, rake, 1>(roadmap.vertices[v], roadmap.vertices[v], environment);
            }

            VAMP_OMP(omp parallel for schedule(runtime) if (parallel))
            for (std::size_t e = 0; e < n_edges; ++e)
            {
                const auto [i, j] = graph.edges[e];
                validity.edges[e] = validity.vertices[i] and validity.vertices[j] and
                                    validate_motion<Robot, rake, resolution>(
                                        roadmap.vertices[i], roadmap.vertices[j], environment);
            }

            return validity;
        }

        // Shortest path over the valid part of a roadmap. Start and goal are each joined to the
        // nearest valid vertices they can reach with a valid motion. Empty path when there is none.
        inline static auto query(
            const Roadmap<Robot> &roadmap,
            const RoadmapGraph &graph,
            const RoadmapValidity &validity,
            const Configuration &start,
            const Configuration &goal,
            const Environment &environment,
            std::size_t max_connections = 8) -> Path<Robot>
        {
            using Link = std::pair<float, std::uint32_t>;  // {distance or cost, vertex}

            Path<Robot> path;
            if (validate_motion<Robot, rake, resolution>(start, goal, environment))
            {
                path.emplace_back(start);
                path.emplace_back(goal);
                return path;
            }

            const std::size_t n = roadmap.vertices.size();

            // ponytail: linear scan over all vertices per query; index the vertices in a kd-tree if
            // queries on very large roadmaps become the bottleneck.
            const auto connect = [&](const Configuration &q, bool from_q) -> std::vector<Link>
            {
                std::vector<Link> nearest;
                for (std::uint32_t v = 0; v < n; ++v)
                {
                    if (validity.vertices[v])
                    {
                        nearest.emplace_back(q.distance(roadmap.vertices[v]), v);
                    }
                }

                const auto keep = std::min(max_connections, nearest.size());
                std::partial_sort(nearest.begin(), nearest.begin() + keep, nearest.end());
                nearest.resize(keep);

                std::vector<Link> links;
                for (const auto &link : nearest)
                {
                    const auto &vertex = roadmap.vertices[link.second];
                    if (from_q ? validate_motion<Robot, rake, resolution>(q, vertex, environment) :
                                 validate_motion<Robot, rake, resolution>(vertex, q, environment))
                    {
                        links.emplace_back(link);
                    }
                }

                return links;
            };

            const auto starts = connect(start, true);
            const auto goals = connect(goal, false);
            if (starts.empty() or goals.empty())
            {
                return path;
            }

            // A* with the goal as virtual vertex n
            constexpr auto none = std::numeric_limits<std::uint32_t>::max();
            std::vector<float> cost(n + 1, std::numeric_limits<float>::infinity());
            std::vector<std::uint32_t> parent(n + 1, none);
            std::vector<char> closed(n + 1, 0);
            std::vector<float> to_goal(n, std::numeric_limits<float>::infinity());
            for (const auto &[distance, v] : goals)
            {
                to_goal[v] = distance;
            }

            std::priority_queue<Link, std::vector<Link>, std::greater<Link>> open;
            for (const auto &[distance, v] : starts)
            {
                cost[v] = distance;
                open.emplace(distance + roadmap.vertices[v].distance(goal), v);
            }

            const auto relax = [&](std::uint32_t from, std::uint32_t to, float length, float heuristic)
            {
                if (cost[from] + length < cost[to])
                {
                    cost[to] = cost[from] + length;
                    parent[to] = from;
                    open.emplace(cost[to] + heuristic, to);
                }
            };

            while (not open.empty())
            {
                const auto current = open.top().second;
                open.pop();
                if (current == n)
                {
                    break;
                }

                if (closed[current])
                {
                    continue;
                }

                closed[current] = 1;
                if (to_goal[current] < std::numeric_limits<float>::infinity())
                {
                    relax(current, n, to_goal[current], 0.F);
                }

                for (auto a = graph.offsets[current]; a < graph.offsets[current + 1]; ++a)
                {
                    const auto [neighbor, edge] = graph.adjacency[a];
                    if (validity.edges[edge] and not closed[neighbor])
                    {
                        relax(
                            current,
                            neighbor,
                            roadmap.vertices[current].distance(roadmap.vertices[neighbor]),
                            roadmap.vertices[neighbor].distance(goal));
                    }
                }
            }

            if (parent[n] == none)
            {
                return path;
            }

            path.emplace_back(goal);
            for (auto v = parent[n]; v != none; v = parent[v])
            {
                path.emplace_back(roadmap.vertices[v]);
            }

            path.emplace_back(start);
            std::reverse(path.begin(), path.end());
            return path;
        }
    };
}  // namespace vamp::planning

#undef VAMP_OMP
