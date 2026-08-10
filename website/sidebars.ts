import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    'intro',
    {
      type: 'category',
      label: 'Getting Started',
      items: [
        'getting-started/installation',
        'getting-started/repo-tour',
        'getting-started/common-commands',
      ],
    },
    {
      type: 'category',
      label: 'Tasks',
      items: [
        'tasks/overview',
        {
          type: 'category',
          label: 'Locomotion',
          link: {
            type: 'doc',
            id: 'tasks/locomotion',
          },
          items: [
            {
              type: 'category',
              label: 'Velocity Locomotion',
              link: {
                type: 'doc',
                id: 'tasks/velocity-locomotion/overview',
              },
              items: [
                {
                  type: 'doc',
                  id: 'tasks/velocity-locomotion/baselines',
                  label: 'Platforms and Baselines',
                },
                {
                  type: 'doc',
                  id: 'tasks/velocity-locomotion/pm01',
                  label: 'EngineAI PM01',
                },
                {
                  type: 'doc',
                  id: 'tasks/g1-brainco-revo',
                  label: 'G1 + BrainCo Revo2',
                },
              ],
            },
            {
              type: 'doc',
              id: 'tasks/lidar-based',
              label: 'LiDAR Locomotion',
            },
          ],
        },
        {
          type: 'category',
          label: 'Motion Imitation',
          link: {
            type: 'doc',
            id: 'tasks/motion-imitation',
          },
          items: ['tasks/motion-tracking/overview'],
        },
        {
          type: 'category',
          label: 'Manipulation',
          link: {
            type: 'doc',
            id: 'tasks/manipulation',
          },
          items: [
            'tasks/loco-manipulation',
            {
              type: 'category',
              label: 'Quadruped + Arm',
              link: {
                type: 'doc',
                id: 'tasks/quadruped-with-arm',
              },
              items: [
                'tasks/quadruped-with-arm/go2-airbot',
                'tasks/quadruped-with-arm/go2-d1-wbc',
                'tasks/quadruped-with-arm/go2-d1-pick-place',
                'tasks/quadruped-with-arm/b2-z1-wbc',
                'tasks/quadruped-with-arm/pick-throw',
                'tasks/quadruped-with-arm/badminton',
              ],
            },
            {
              type: 'doc',
              id: 'tasks/revo3-dexhand',
              label: 'Revo3 Dexterous Hand',
            },
          ],
        },
        'tasks/multi-agent',
        {
          type: 'category',
          label: 'Planning & Autonomy',
          link: {
            type: 'doc',
            id: 'tasks/planning-autonomy',
          },
          items: ['tasks/llm-high-level-planning'],
        },
      ],
    },
    {
      type: 'category',
      label: 'Training & Methods',
      link: {
        type: 'doc',
        id: 'training-methods/overview',
      },
      items: [
        {
          type: 'category',
          label: 'Student-Teacher Learning',
          link: {
            type: 'doc',
            id: 'tasks/student-teacher/overview',
          },
          items: [
            'tasks/student-teacher/depth-backbones',
            'tasks/student-teacher/losses',
            'tasks/student-teacher/apex-distillation',
            'tasks/student-teacher/concurrent-teacher-student',
            {
              type: 'doc',
              id: 'tasks/velocity-locomotion/depth-students',
              label: 'Velocity Student Variants',
            },
            'tasks/student-teacher/training-recipes',
            'tasks/student-teacher/sim2sim',
            'tasks/student-teacher/debugging',
          ],
        },
        {
          type: 'category',
          label: 'Locomotion Methods',
          link: {
            type: 'doc',
            id: 'training-methods/locomotion',
          },
          items: [
            'tasks/velocity-locomotion/start',
            'tasks/velocity-locomotion/real',
            'tasks/velocity-locomotion/ame',
            'tasks/velocity-locomotion/contact-trails',
          ],
        },
        {
          type: 'doc',
          id: 'training-methods/score-matching-motion-priors',
          label: 'Score-Matching Motion Priors',
        },
        {
          type: 'category',
          label: 'Motion Tracking Methods',
          link: {
            type: 'doc',
            id: 'training-methods/motion-tracking',
          },
          items: [
            {
              type: 'category',
              label: 'APEX',
              link: {
                type: 'doc',
                id: 'tasks/go2-apex',
              },
              items: [
                'tasks/apex/training',
                'tasks/apex/motion-data',
                'tasks/apex/sim2sim',
                'tasks/apex/sim2real',
              ],
            },
            {
              type: 'category',
              label: 'BeyondMimic',
              link: {
                type: 'doc',
                id: 'tasks/beyondmimic/overview',
              },
              items: [
                'tasks/beyondmimic/g1',
                'tasks/beyondmimic/pm01',
                'tasks/beyondmimic/t800',
              ],
            },
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      items: [
        'reference/credits',
        'reference/task-registry',
        'reference/task-status',
        'reference/cli-reference',
        'reference/deployment-artifacts',
      ],
    },
  ],
};

export default sidebars;
