#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <vamp/planning/pprm.hh>
#include <vamp/random/halton.hh>

#if defined(__x86_64__)
#include <vamp/random/xorshift.hh>
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "planning_common.hh"

using namespace vamp::test;

namespace
{
    using PPRM = vamp::planning::PPRM<Robot, rake, Robot::resolution>;
    using RNG = vamp::rng::RNG<Robot>;
    using MakeRNG = std::function<RNG::Ptr()>;

    auto halton(std::size_t skip) -> MakeRNG
    {
        return [skip]() -> RNG::Ptr
        {
            auto rng = std::make_shared<vamp::rng::Halton<Robot>>();
            for (std::size_t i = 0; i < skip; ++i)
            {
                rng->next();
            }

            return rng;
        };
    }

    auto seeds() -> std::vector<MakeRNG>
    {
        std::vector<MakeRNG> result{halton(0), halton(1000), halton(54321)};
#if defined(__x86_64__)
        result.emplace_back([]() -> RNG::Ptr { return std::make_shared<vamp::rng::XORShift<Robot>>(); });
        result.emplace_back(
            []() -> RNG::Ptr { return std::make_shared<vamp::rng::XORShift<Robot>>(11UL, 7UL); });
#endif
        return result;
    }

    // Thread counts to try; a single entry when built without OpenMP.
    auto thread_counts() -> std::vector<int>
    {
#if defined(_OPENMP)
        return {1, 2, 4, omp_get_num_procs()};
#else
        return {1};
#endif
    }

    void set_threads([[maybe_unused]] int threads)
    {
#if defined(_OPENMP)
        omp_set_num_threads(threads);
#endif
    }

    // True when the batched roadmap equals the serial one and both left the sampler in the same state.
    auto equivalent(
        const Environment &environment,
        const Settings &config,
        const MakeRNG &make_rng,
        const Roadmap &serial) -> bool
    {
        auto rng = make_rng();
        const auto parallel = PPRM::build_roadmap(start(), goal(), environment, config, rng);

        // The serial planner drew iterations - 1 samples
        auto replay = make_rng();
        for (std::size_t i = 0; i + 1 < serial.iterations; ++i)
        {
            replay->next();
        }

        return same_vertices(serial, parallel) and edge_set(serial) == edge_set(parallel) and
               serial.iterations == parallel.iterations and same(rng->next(), replay->next());
    }

    auto serial_validity(const Roadmap &roadmap, const Environment &environment)
        -> vamp::planning::RoadmapValidity
    {
        vamp::planning::RoadmapValidity validity;
        for (const auto &vertex : roadmap.vertices)
        {
            validity.vertices.emplace_back(valid(vertex, environment));
        }

        for (std::size_t j = 0; j < roadmap.edges.size(); ++j)
        {
            for (const auto i : roadmap.edges[j])
            {
                if (i < j)
                {
                    validity.edges.emplace_back(
                        validity.vertices[i] and validity.vertices[j] and
                        valid(roadmap.vertices[i], roadmap.vertices[j], environment));
                }
            }
        }

        return validity;
    }
}  // namespace

TEST_CASE("PPRM roadmap equals the PRM roadmap for any batch size", "[planning][pprm]")
{
    set_threads(1);
    const std::vector<Environment> environments{Environment(), sphere_cage()};
    for (const auto &environment : environments)
    {
        for (const auto &make_rng : seeds())
        {
            auto config = settings(1500);
            const auto serial = PRM::build_roadmap(start(), goal(), environment, config, make_rng());
            REQUIRE(serial.vertices.size() == 1500);

            for (const std::size_t batch_size : {1, 2, 7, 64, 1000, 5000})
            {
                config.batch_size = batch_size;
                CAPTURE(batch_size);
                REQUIRE(equivalent(environment, config, make_rng, serial));
            }
        }
    }
}

TEST_CASE("PPRM roadmap equals the PRM roadmap at the iteration limit", "[planning][pprm]")
{
    set_threads(1);
    const auto environment = sphere_cage();
    const auto make_rng = halton(0);
    auto config = settings(1500);
    config.max_iterations = 777;
    const auto serial = PRM::build_roadmap(start(), goal(), environment, config, make_rng());
    REQUIRE(serial.vertices.size() < 777);

    for (const std::size_t batch_size : {1, 64, 5000})
    {
        config.batch_size = batch_size;
        CAPTURE(batch_size);
        REQUIRE(equivalent(environment, config, make_rng, serial));
    }
}

TEST_CASE("PPRM roadmap does not depend on the thread count or schedule", "[planning][pprm]")
{
    const auto environment = sphere_cage();
    const auto make_rng = halton(0);
    auto config = settings(1500);
    config.batch_size = 256;
    const auto serial = PRM::build_roadmap(start(), goal(), environment, config, make_rng());

    for (const auto threads : thread_counts())
    {
        set_threads(threads);
        CAPTURE(threads);
#if defined(_OPENMP)
        for (const auto schedule : {omp_sched_static, omp_sched_dynamic, omp_sched_guided})
        {
            omp_set_schedule(schedule, 0);
            REQUIRE(equivalent(environment, config, make_rng, serial));
        }
#else
        REQUIRE(equivalent(environment, config, make_rng, serial));
#endif
    }
}

TEST_CASE("Parallel edge validation agrees with serial validation", "[planning][pprm]")
{
    // Edges of a roadmap built in free space, checked against the cage: a mix of valid and invalid
    const auto cage = sphere_cage();
    const auto roadmap = PRM::build_roadmap(start(), goal(), Environment(), settings(1000), halton(0)());
    const auto edges = edge_set(roadmap);
    const auto endpoints = [&](std::size_t e) -> std::pair<Configuration, Configuration>
    { return {roadmap.vertices[edges[e].first], roadmap.vertices[edges[e].second]}; };

    std::vector<char> expected;
    for (const auto &[i, j] : edges)
    {
        expected.emplace_back(valid(roadmap.vertices[i], roadmap.vertices[j], cage));
    }

    const auto n_valid = static_cast<std::size_t>(std::count(expected.begin(), expected.end(), 1));
    REQUIRE(n_valid > 0);
    REQUIRE(n_valid < edges.size());

    for (const auto threads : thread_counts())
    {
        set_threads(threads);
        CAPTURE(threads);
        std::vector<char> actual;
        PPRM::validate_edges(edges.size(), endpoints, cage, actual);
        REQUIRE(actual == expected);
    }
}

TEST_CASE("Revalidation equals a serial recheck after the obstacles move", "[planning][pprm]")
{
    const auto before = sphere_cage();
    const auto roadmap = PRM::build_roadmap(start(), goal(), before, settings(1500), halton(0)());
    const auto graph = PPRM::graph(roadmap);
    REQUIRE(graph.edges.size() == edge_set(roadmap).size());
    REQUIRE(graph.adjacency.size() == 2 * graph.edges.size());

    for (const auto threads : thread_counts())
    {
        set_threads(threads);
        CAPTURE(threads);

        // Nothing moved: the whole roadmap is still valid
        const auto unmoved = PPRM::revalidate(roadmap, graph, before);
        REQUIRE(std::count(unmoved.vertices.begin(), unmoved.vertices.end(), 1) == 1500);
        REQUIRE(
            static_cast<std::size_t>(std::count(unmoved.edges.begin(), unmoved.edges.end(), 1)) ==
            graph.edges.size());

        for (const float shift : {0.05F, 0.2F})
        {
            const auto after = sphere_cage(shift);
            const auto validity = PPRM::revalidate(roadmap, graph, after);
            const auto expected = serial_validity(roadmap, after);
            REQUIRE(validity.vertices == expected.vertices);
            REQUIRE(validity.edges == expected.edges);
            REQUIRE(std::count(validity.edges.begin(), validity.edges.end(), 0) > 0);
        }
    }
}

TEST_CASE("Paths found on a revalidated roadmap are valid", "[planning][pprm]")
{
    set_threads(thread_counts().back());
    const auto roadmap = PRM::build_roadmap(start(), goal(), sphere_cage(), settings(3000), halton(0)());
    const auto graph = PPRM::graph(roadmap);

    for (const float shift : {0.F, 0.02F, 0.05F})
    {
        CAPTURE(shift);
        const auto environment = sphere_cage(shift);
        REQUIRE(valid(start(), environment));
        REQUIRE(valid(goal(), environment));

        const auto validity = PPRM::revalidate(roadmap, graph, environment);
        const auto path = PPRM::query(roadmap, graph, validity, start(), goal(), environment);
        REQUIRE(path.size() > 2);
        REQUIRE(same(path.front(), start()));
        REQUIRE(same(path.back(), goal()));

        // Recheck with the original validator, edge by edge
        bool collision_free = true;
        for (std::size_t i = 0; i + 1 < path.size(); ++i)
        {
            collision_free = collision_free and valid(path[i], path[i + 1], environment);
        }

        REQUIRE(collision_free);
    }
}

TEST_CASE("Queries on a revalidated roadmap handle the trivial and the blocked case", "[planning][pprm]")
{
    const auto roadmap = PRM::build_roadmap(start(), goal(), sphere_cage(), settings(500), halton(0)());
    const auto graph = PPRM::graph(roadmap);

    // Free space: the straight line is valid
    const Environment free_space;
    const auto direct = PPRM::query(
        roadmap, graph, PPRM::revalidate(roadmap, graph, free_space), start(), goal(), free_space);
    REQUIRE(direct.size() == 2);

    // A sphere that swallows the robot: nothing is valid
    vamp::collision::Environment<float> blocked_input;
    blocked_input.spheres.emplace_back(vamp::collision::factory::sphere::flat(0, 0, 0.5, 2.0));
    const Environment blocked(blocked_input);
    const auto validity = PPRM::revalidate(roadmap, graph, blocked);
    REQUIRE(std::count(validity.vertices.begin(), validity.vertices.end(), 1) == 0);
    REQUIRE(std::count(validity.edges.begin(), validity.edges.end(), 1) == 0);
    REQUIRE(PPRM::query(roadmap, graph, validity, start(), goal(), blocked).empty());
}

// Small on purpose: this is the case to run under -fsanitize=thread.
TEST_CASE("PPRM construction and revalidation with several threads", "[planning][pprm][race]")
{
    set_threads(4);
    const auto environment = sphere_cage();
    const auto make_rng = halton(0);
    auto config = settings(600);
    config.batch_size = 64;
    const auto serial = PRM::build_roadmap(start(), goal(), environment, config, make_rng());
    REQUIRE(equivalent(environment, config, make_rng, serial));

    const auto graph = PPRM::graph(serial);
    const auto moved = sphere_cage(0.05F);
    const auto validity = PPRM::revalidate(serial, graph, moved);
    REQUIRE(validity.edges == serial_validity(serial, moved).edges);
}
