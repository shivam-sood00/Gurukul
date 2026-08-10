import Link from '@docusaurus/Link';

import styles from './WelcomeContent.module.css';

const taskSections = [
  {
    title: 'Locomotion',
    paths: 'Velocity tracking · LiDAR locomotion',
    description: 'Choose a movement objective, terrain contract, and supported robot platform.',
    to: '/docs/tasks/locomotion',
  },
  {
    title: 'Motion Imitation',
    paths: 'Reference motion · motion style',
    description: 'Make Go2, G1, PM01, T800, or B2+Z1 follow reference motion.',
    to: '/docs/tasks/motion-imitation',
  },
  {
    title: 'Manipulation',
    paths: 'Mobile manipulation · dexterous hands',
    description: 'Select whole-body object interaction, quadruped-with-arm, or Revo3 tasks.',
    to: '/docs/tasks/manipulation',
  },
  {
    title: 'Multi-Agent',
    paths: 'Coordination · hierarchical control',
    description: 'Train Go2 + B2 DirectMARL collaboration and high-level velocity coordination.',
    to: '/docs/tasks/multi-agent',
  },
  {
    title: 'Planning & Autonomy',
    paths: 'High-level decisions · physical feedback',
    description: 'Evaluate language-model planning over bounded robot-control primitives.',
    to: '/docs/tasks/planning-autonomy',
  },
];

const methodSections = [
  {
    title: 'Student-Teacher Learning',
    paths: 'Teachers · students · distillation',
    description: 'Train privileged teachers, deployable students, depth policies, and action priors.',
    to: '/docs/tasks/student-teacher/overview',
  },
  {
    title: 'Locomotion Methods',
    paths: 'START · REAL · AME · Contact Trails',
    description: 'Compare locomotion learning methods and policy-side terrain representations.',
    to: '/docs/training-methods/locomotion',
  },
  {
    title: 'Score-Matching Motion Priors',
    paths: 'Diffusion priors · motion-guided rewards',
    description: 'Pretrain morphology-specific motion priors and use them to guide velocity-policy learning.',
    to: '/docs/training-methods/score-matching-motion-priors',
  },
  {
    title: 'Motion Tracking Methods',
    paths: 'APEX · BeyondMimic',
    description: 'Find motion data, training, and method-specific Sim2Sim or Sim2Real guidance.',
    to: '/docs/training-methods/motion-tracking',
  },
];

export default function WelcomeContent(): JSX.Element {
  return (
    <div className={styles.welcome}>
      <section className={styles.intro}>
        <div className={styles.introText}>
          <p className={styles.eyebrow}>Robot-learning research in Isaac Lab</p>
          <h1 className={styles.title}>Gurukul</h1>
          <p className={styles.lede}>
            Gurukul brings together implementations of my own research and a growing collection of other exciting work
            from the robotics community. It spans legged locomotion, motion imitation, manipulation, perception, and
            multi-robot learning.
          </p>
          <p>
            The goal is to keep everything in one place: making it easier to build on existing research while creating
            a connected knowledge base that LLM agents can navigate, combine, and use to develop new work. The project
            is still in its early stages and will grow over time with more tasks, methods, and research.
          </p>
          <div className={styles.researchNote}>
            Implemented tasks are tagged by their current stage, distinguishing completed baselines from work that is
            still under development. See the <Link to="/docs/reference/task-status">Task Status</Link> catalog for
            details.
          </div>
        </div>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2>Tasks</h2>
          <p>Start with what you want the robot to do.</p>
        </div>
        <div className={styles.taskGrid}>
          {taskSections.map((section) => (
            <article className={styles.taskCard} key={section.to}>
              <h3>{section.title}</h3>
              <span className={styles.taskMeta}>{section.paths}</span>
              <p>{section.description}</p>
              <Link to={section.to}>Open section</Link>
            </article>
          ))}
        </div>
      </section>

      <section className={styles.section}>
        <div className={styles.sectionHeader}>
          <h2>Training &amp; Methods</h2>
          <p>Choose how the task policy is trained, represented, distilled, and transferred.</p>
        </div>
        <div className={styles.taskGrid}>
          {methodSections.map((section) => (
            <article className={styles.taskCard} key={section.to}>
              <h3>{section.title}</h3>
              <span className={styles.taskMeta}>{section.paths}</span>
              <p>{section.description}</p>
              <Link to={section.to}>Open section</Link>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}
