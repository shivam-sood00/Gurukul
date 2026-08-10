import clsx from 'clsx';

import styles from './StatusBadges.module.css';

type Maturity = 'validated' | 'maintained' | 'experimental' | 'legacy';
type Target = 'sim' | 'sim2sim' | 'sim2real' | 'hardware';

const maturityLabels: Record<Maturity, {label: string; title: string}> = {
  validated: {
    label: 'Validated',
    title: 'Runnable path has been checked end to end for the listed targets.',
  },
  maintained: {
    label: 'Maintained',
    title: 'Expected to work and kept current, but not fully validated across transfer or deploy paths.',
  },
  experimental: {
    label: 'Experimental',
    title: 'Prototype or research path that may change and may require manual debugging.',
  },
  legacy: {
    label: 'Legacy',
    title: 'Kept for reference and not actively maintained.',
  },
};

const targetLabels: Record<Target, string> = {
  sim: 'Sim',
  sim2sim: 'Sim2Sim',
  sim2real: 'Sim2Real',
  hardware: 'Hardware',
};

type StatusBadgesProps = {
  maturity: Maturity;
  targets?: Target[];
  note?: string;
};

export default function StatusBadges({maturity, targets = [], note}: StatusBadgesProps) {
  const maturityMeta = maturityLabels[maturity];

  return (
    <div className={styles.statusBadges} aria-label="Documentation status">
      <span className={clsx(styles.badge, styles[maturity])} title={maturityMeta.title}>
        {maturityMeta.label}
      </span>
      {targets.map((target) => (
        <span className={clsx(styles.badge, styles.target)} key={target}>
          {targetLabels[target]}
        </span>
      ))}
      {note ? <span className={styles.note}>{note}</span> : null}
    </div>
  );
}
