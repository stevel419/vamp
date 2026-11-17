#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <vamp/random/halton.hh>

#include "planning_common.hh"

using namespace vamp::test;

namespace
{
    auto build(const Environment &environment, const Settings &settings) -> Roadmap
    {
        return PRM::build_roadmap(
            start(), goal(), environment, settings, std::make_shared<vamp::rng::Halton<Robot>>());
    }
}  // namespace

TEST_CASE("PRM roadmap is the same for the same sampler state", "[planning][prm]")
{
    const auto environment = sphere_cage();
    const auto a = build(environment, settings(400));
    const auto b = build(environment, settings(400));

    REQUIRE(same_vertices(a, b));
    REQUIRE(a.edges == b.edges);
    REQUIRE(a.iterations == b.iterations);
}

TEST_CASE("PRM roadmap vertices are start, goal, then the valid samples in order", "[planning][prm]")
{
    const auto environment = sphere_cage();
    const auto roadmap = build(environment, settings(400));

    REQUIRE(roadmap.vertices.size() == 400);
    REQUIRE(roadmap.edges.size() == roadmap.vertices.size());
    REQUIRE(same(roadmap.vertices[0], start()));
    REQUIRE(same(roadmap.vertices[1], goal()));

    vamp::rng::Halton<Robot> replay;
    std::size_t draws = 0;
    bool in_order = true;
    for (std::size_t v = 2; v < roadmap.vertices.size(); ++v)
    {
        auto sample = replay.next();
        ++draws;
        while (not valid(sample, environment))
        {
            sample = replay.next();
            ++draws;
        }

        in_order = in_order and same(sample, roadmap.vertices[v]);
    }

    REQUIRE(in_order);
    // The planner counts one more iteration than it draws samples
    REQUIRE(roadmap.iterations == draws + 1);
}

TEST_CASE("PRM roadmap stops at the iteration limit", "[planning][prm]")
{
    const auto environment = sphere_cage();
    auto limited = settings(400);
    limited.max_iterations = 50;
    const auto roadmap = build(environment, limited);

    vamp::rng::Halton<Robot> replay;
    std::size_t n_valid = 0;
    for (auto i = 0U; i < limited.max_iterations; ++i)
    {
        n_valid += valid(replay.next(), environment);
    }

    REQUIRE(n_valid > 0);
    REQUIRE(n_valid < limited.max_iterations);
    REQUIRE(roadmap.vertices.size() == n_valid + 2);
    REQUIRE(roadmap.iterations == limited.max_iterations + 1);
}

TEST_CASE("PRM roadmap is a simple undirected graph of valid motions", "[planning][prm]")
{
    const auto environment = sphere_cage();
    const auto roadmap = build(environment, settings(400));

    bool simple = true;
    bool symmetric = true;
    bool collision_free = true;
    std::size_t n_edges = 0;
    for (std::size_t j = 0; j < roadmap.edges.size(); ++j)
    {
        const auto &neighbors = roadmap.edges[j];
        const std::set<std::size_t> unique(neighbors.begin(), neighbors.end());
        simple = simple and unique.size() == neighbors.size() and unique.count(j) == 0;

        for (const auto i : neighbors)
        {
            const auto &back = roadmap.edges[i];
            symmetric = symmetric and std::find(back.begin(), back.end(), j) != back.end();
            if (i < j)
            {
                ++n_edges;
                collision_free =
                    collision_free and valid(roadmap.vertices[i], roadmap.vertices[j], environment);
            }
        }
    }

    REQUIRE(n_edges > 0);
    REQUIRE(simple);
    REQUIRE(symmetric);
    REQUIRE(collision_free);
}

TEST_CASE("PRM connects each vertex to its valid k-nearest earlier vertices", "[planning][prm]")
{
    const auto environment = sphere_cage();
    const auto config = settings(400);
    const auto roadmap = build(environment, config);

    bool matches = true;
    for (std::size_t j = 2; j < roadmap.vertices.size(); ++j)
    {
        // Vertex j was connected when the roadmap held the j vertices before it
        const auto k = config.max_neighbors(j);
        const auto r = config.neighbor_radius(j);

        std::vector<std::pair<float, std::size_t>> nearest;
        for (std::size_t i = 0; i < j; ++i)
        {
            const auto distance = (roadmap.vertices[i] - roadmap.vertices[j]).l2_norm();
            if (distance <= r)
            {
                nearest.emplace_back(distance, i);
            }
        }

        std::sort(nearest.begin(), nearest.end());
        nearest.resize(std::min(k, nearest.size()));

        std::set<std::size_t> expected;
        for (const auto &[distance, i] : nearest)
        {
            if (valid(roadmap.vertices[i], roadmap.vertices[j], environment))
            {
                expected.emplace(i);
            }
        }

        std::set<std::size_t> actual;
        for (const auto i : roadmap.edges[j])
        {
            if (i < j)
            {
                actual.emplace(i);
            }
        }

        matches = matches and actual == expected;
    }

    REQUIRE(matches);
}
