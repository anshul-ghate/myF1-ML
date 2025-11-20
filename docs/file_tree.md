```
f1_fan_analytics_and_prediction_app/
├── .github/
│   └── workflows/
│       └── ci.yml         # Continuous integration pipeline
├── docs/
│   ├── prd_f1_fan_analytics_app.md
│   ├── system_design.md
│   ├── architect.plantuml
│   ├── class_diagram.plantuml
│   ├── sequence_diagram.plantuml
│   ├── er_diagram.plantuml
│   └── ui_navigation.plantuml
├── public/
│   ├── images/
│   └── favicon.ico
├── src/
│   ├── app/
│   │   ├── (main)/
│   │   │   ├── dashboard/
│   │   │   │   └── page.tsx
│   │   │   ├── live-race/[season]/[round]/
│   │   │   │   └── page.tsx
│   │   │   ├── strategy-sandbox/[season]/[round]/
│   │   │   │   └── page.tsx
│   │   │   └── settings/
│   │   │       └── page.tsx
│   │   ├── layout.tsx
│   │   └── globals.css
│   ├── components/
│   │   ├── ui/              # Shadcn-ui components
│   │   ├── common/
│   │   │   ├── Header.tsx
│   │   │   └── Footer.tsx
│   │   ├── dashboard/
│   │   │   ├── StandingsWidget.tsx
│   │   │   └── UpcomingRaceWidget.tsx
│   │   ├── live/
│   │   │   ├── LiveLeaderboard.tsx
│   │   │   └── EventFeed.tsx
│   │   └── assistant/
│   │       └── ChatWindow.tsx
│   ├── services/
│   │   ├── dataService.ts   # Interacts with LocalStorage and F1 API
│   │   ├── aiService.ts     # Interacts with Generative AI API
│   │   └── simulatorService.ts # Logic for strategy simulation
│   ├── hooks/
│   │   ├── useUserProfile.ts
│   │   └── useRaceData.ts
│   ├── store/
│   │   └── useAppStore.ts   # Zustand store for global state
│   ├── types/
│   │   └── index.ts         # TypeScript type definitions
│   └── lib/
│       └── utils.ts         # Utility functions
├── .eslintrc.json
├── .gitignore
├── next.config.mjs
├── package.json
├── postcss.config.js
├── tailwind.config.ts
└── tsconfig.json
```