#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <vamp/collision/factory.hh>
#include <vamp/planning/plan.hh>
#include <vamp/planning/prm.hh>
#include <vamp/planning/roadmap.hh>
#include <vamp/planning/validate.hh>
#include <vamp/robots/panda.hh>
#include <vamp/vector.hh>

// Shared fixtures for the roadmap planner tests: the sphere cage problem of
// scripts/sphere_cage_example.py and helpers to compare roadmaps.
namespace vamp::test
{
    using Robot = vamp::robots::Panda;
    inline constexpr std::size_t rake = vamp::FloatVectorWidth;
    using Configuration = Robot::Configuration;
    using Environment = vamp::collision::Environment<vamp::FloatVector<rake>>;
    using Roadmap = vamp::planning::Roadmap<Robot>;
    using Settings = vamp::planning::RoadmapSettings<vamp::planning::PRMStarNeighborParams>;
    using PRM = vamp::planning::PRM<Robot, rake, Robot::resolution>;
    using Edges = std::vector<std::pair<std::size_t, std::size_t>>;

    inline auto start() -> Configuration
    {
        return Configuration(Robot::ConfigurationArray{0., -0.785, 0., -2.356, 0., 1.571, 0.785});
    }

    inline auto goal() -> Configuration
    {
        return Configuration(Robot::ConfigurationArray{2.35, 1., 0., -0.8, 0, 2.5, 0.785});
    }

    // Fourteen spheres around the robot, optionally moved along x.
    inline auto sphere_cage(float shift = 0.F, float radius = 0.2F) -> Environment
    {
        static const std::vector<std::array<float, 3>> centers = {
            {0.55, 0, 0.25},
            {0.35, 0.35, 0.25},
            {0, 0.55, 0.25},
            {-0.55, 0, 0.25},
            {-0.35, -0.35, 0.25},
            {0, -0.55, 0.25},
            {0.35, -0.35, 0.25},
            {0.35, 0.35, 0.8},
            {0, 0.55, 0.8},
            {-0.35, 0.35, 0.8},
            {-0.55, 0, 0.8},
            {-0.35, -0.35, 0.8},
            {0, -0.55, 0.8},
            {0.35, -0.35, 0.8},
        };

        vamp::collision::Environment<float> environment;
        for (const auto &center : centers)
        {
            environment.spheres.emplace_back(
                vamp::collision::factory::sphere::flat(center[0] + shift, center[1], center[2], radius));
        }

        environment.sort();
        return Environment(environment);
    }

    inline auto settings(std::size_t max_samples) -> Settings
    {
        Settings result(vamp::planning::PRMStarNeighborParams(Robot::dimension, Robot::space_measure()));
        result.max_samples = max_samples;
        return result;
    }

    inline auto same(const Configuration &a, const Configuration &b) -> bool
    {
        return a.to_array() == b.to_array();
    }

    inline auto valid(const Configuration &c, const Environment &environment) -> bool
    {
        return vamp::planning::validate_motion<Robot, rake, 1>(c, c, environment);
    }

    inline auto valid(const Configuration &a, const Configuration &b, const Environment &environment) -> bool
    {
        return vamp::planning::validate_motion<Robot, rake, Robot::resolution>(a, b, environment);
    }

    // Every undirected edge once, as (lower, higher), sorted.
    inline auto edge_set(const Roadmap &roadmap) -> Edges
    {
        Edges edges;
        for (std::size_t j = 0; j < roadmap.edges.size(); ++j)
        {
            for (const auto i : roadmap.edges[j])
            {
                if (i < j)
                {
                    edges.emplace_back(i, j);
                }
            }
        }

        std::sort(edges.begin(), edges.end());
        return edges;
    }

    inline auto same_vertices(const Roadmap &a, const Roadmap &b) -> bool
    {
        return a.vertices.size() == b.vertices.size() and
               std::equal(a.vertices.begin(), a.vertices.end(), b.vertices.begin(), same);
    }
}  // namespace vamp::test
