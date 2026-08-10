import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type {Options as PresetOptions} from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Gurukul',
  tagline: 'Robot-learning research in Isaac Lab: locomotion, motion tracking, manipulation, and transfer.',
  favicon: 'img/favicon.svg',

  url: 'https://shivamsood.org',
  baseUrl: '/Gurukul/',
  organizationName: 'shivam-sood00',
  projectName: 'Gurukul',
  deploymentBranch: 'gh-pages',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'docs',
          editUrl: 'https://github.com/shivam-sood00/Gurukul/tree/main/website/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies PresetOptions,
    ],
  ],

  plugins: [
    [
      '@easyops-cn/docusaurus-search-local',
      {
        hashed: true,
        indexDocs: true,
        indexPages: true,
        indexBlog: false,
        language: ['en'],
      },
    ],
  ],

  themeConfig: {
    image: 'img/social-card.svg',
    metadata: [
      {
        name: 'description',
        content:
          'Gurukul is an Isaac Lab research workspace for locomotion, motion tracking, manipulation, and policy transfer.',
      },
    ],
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Gurukul',
      logo: {
        alt: 'Gurukul logo',
        src: 'img/logo.svg',
      },
      items: [
        {to: '/docs/intro', label: 'Docs', position: 'left'},
        {to: '/docs/tasks/overview', label: 'Tasks', position: 'left'},
        {to: '/docs/training-methods/overview', label: 'Training & Methods', position: 'left'},
        {
          href: 'https://github.com/shivam-sood00/Gurukul',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {label: 'Get Started', to: '/docs/getting-started/installation'},
            {label: 'Tasks', to: '/docs/tasks/overview'},
            {label: 'Training & Methods', to: '/docs/training-methods/overview'},
          ],
        },
        {
          title: 'Project',
          items: [
            {label: 'Repository', href: 'https://github.com/shivam-sood00/Gurukul'},
            {label: 'Acknowledgements', to: '/docs/reference/credits'},
            {label: 'License', href: 'https://github.com/shivam-sood00/Gurukul/blob/main/LICENSE'},
          ],
        },
      ],
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'yaml'],
    },
  },
};

export default config;
