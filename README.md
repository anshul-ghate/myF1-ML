# F1 Fan Analytics Platform

A comprehensive Formula 1 analytics platform featuring real-time data, AI-powered predictions, and an intelligent assistant for F1 fans.

## 🏎️ Features

### Core Features
- **Real-time F1 Data**: Live driver and constructor standings, race results, and schedules
- **AI-Powered Predictions**: Machine learning predictions for upcoming races
- **Intelligent Assistant**: OpenAI-powered chatbot for F1 insights and analysis
- **User Authentication**: Secure login with Supabase Auth
- **Personalization**: Save favorite drivers and constructors
- **Responsive Design**: Beautiful UI with dark mode support

### Technical Features
- **Backend**: Supabase (PostgreSQL database, Edge Functions, Authentication)
- **Frontend**: React + TypeScript + Shadcn-UI + Tailwind CSS
- **Data Source**: Ergast F1 API integration
- **Real-time Updates**: Automatic data synchronization
- **ML Predictions**: Historical performance-based predictions

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and pnpm installed
- Supabase account (already configured)

### Installation

1. **Install Dependencies**
```bash
pnpm install
```

2. **Initialize Backend Data**

Navigate to `/admin` page in your browser after starting the dev server, then:
- Click "Sync F1 Data" to populate the database (takes 30-60 seconds)
- Click "Generate Predictions" to create race predictions

3. **Start Development Server**
```bash
pnpm run dev
```

The application will be available at `http://localhost:5173`

### Building for Production
```bash
pnpm run build
```

## 📊 Backend Architecture

### Database Schema

The application uses 8 main tables:

1. **app_b64c9980ff_drivers**: Driver information
2. **app_b64c9980ff_constructors**: Team information
3. **app_b64c9980ff_circuits**: Circuit details
4. **app_b64c9980ff_seasons**: Season data
5. **app_b64c9980ff_races**: Race schedule
6. **app_b64c9980ff_race_results**: Race outcomes
7. **app_b64c9980ff_predictions**: ML predictions
8. **app_b64c9980ff_profiles**: User profiles

### Edge Functions

Three serverless functions power the backend:

1. **app_b64c9980ff_ai_assistant**: OpenAI-powered F1 assistant
   - Endpoint: `/functions/v1/app_b64c9980ff_ai_assistant`
   - Requires: `OPENAI_API_KEY` environment variable

2. **app_b64c9980ff_data_sync**: Syncs F1 data from Ergast API
   - Endpoint: `/functions/v1/app_b64c9980ff_data_sync`
   - Run after each race to update results

3. **app_b64c9980ff_generate_predictions**: Generates race predictions
   - Endpoint: `/functions/v1/app_b64c9980ff_generate_predictions`
   - Run before each race weekend

## 🔧 Configuration

### Supabase Configuration

The application is pre-configured with:
- **Project URL**: `https://hprhbsgmjjjgojkdasay.supabase.co`
- **Project REF**: `hprhbsgmjjjgojkdasay`
- **Anon Key**: Already set in `src/lib/supabaseClient.ts`

### Environment Variables (Edge Functions)

The following environment variables are required for edge functions:

```bash
# Required for AI Assistant
OPENAI_API_KEY=your_openai_api_key

# Auto-configured by Supabase
SUPABASE_URL=https://hprhbsgmjjjgojkdasay.supabase.co
SUPABASE_SERVICE_ROLE_KEY=auto_configured
```

To set the OpenAI API key:
1. Go to Supabase Dashboard → Edge Functions
2. Select `app_b64c9980ff_ai_assistant`
3. Add `OPENAI_API_KEY` in the Secrets section

## 📱 Usage Guide

### For End Users

1. **Dashboard**: View current standings, race countdown, and last race results
2. **Schedule**: Check upcoming race dates and times
3. **AI Assistant**: Ask questions about F1 (requires OpenAI API key setup)
4. **Profile**: Set favorite driver and constructor (requires login)

### For Administrators

Access the admin panel at `/admin` to:
- Sync latest F1 data from Ergast API
- Generate predictions for upcoming races
- Monitor backend operations

**Recommended Maintenance Schedule**:
- Run data sync after each race (Sunday evening)
- Generate predictions before each race weekend (Thursday)
- Optional: Daily data sync for any updates

## 🏗️ Project Structure

```
src/
├── components/
│   ├── auth/              # Authentication components
│   ├── dashboard/         # Dashboard widgets
│   ├── details/           # Detail view components
│   ├── layout/            # Layout components
│   ├── skeletons/         # Loading skeletons
│   └── ui/                # Shadcn-UI components
├── contexts/
│   └── AuthContext.tsx    # Authentication context
├── data/
│   └── types.ts           # TypeScript type definitions
├── hooks/
│   ├── usePagination.ts   # Pagination hook
│   └── useSortableData.ts # Sorting hook
├── lib/
│   ├── backendDataService.ts      # Backend data layer
│   ├── dataFetchingService.ts     # Data fetching service
│   ├── supabaseClient.ts          # Supabase client
│   ├── supabaseService.ts         # Supabase operations
│   └── setupBackend.ts            # Backend initialization
└── pages/
    ├── Index.tsx          # Main dashboard
    ├── AdminPage.tsx      # Admin panel
    ├── LoginPage.tsx      # Login page
    ├── SignupPage.tsx     # Signup page
    ├── ProfilePage.tsx    # User profile
    └── SettingsPage.tsx   # User settings
```

## 🔐 Security Features

- Row Level Security (RLS) enabled on all tables
- Public read access for F1 data
- User-specific access for profiles
- JWT-based authentication
- Secure edge function endpoints

## 🎨 UI Features

- **Dark Mode**: Automatic theme switching
- **Responsive Design**: Mobile, tablet, and desktop support
- **Custom Scrollbars**: Themed scrollbars matching the design
- **Loading States**: Skeleton loaders for better UX
- **Interactive Tables**: Sortable and paginated data views
- **Modal Dialogs**: Detailed views for drivers and constructors

## 📈 Data Flow

1. **Data Ingestion**: Edge function fetches data from Ergast API
2. **Storage**: Data stored in Supabase PostgreSQL
3. **Predictions**: ML model generates predictions based on historical data
4. **Frontend**: React components fetch and display data
5. **Real-time**: Supabase provides real-time updates (optional)

## 🤖 AI Assistant

The AI assistant uses OpenAI's GPT-4o-mini model to provide:
- Race analysis and insights
- Driver and team comparisons
- Historical context and statistics
- Personalized responses based on user favorites

## 🐛 Troubleshooting

### No Data Showing
- Visit `/admin` and run "Sync F1 Data"
- Check browser console for errors
- Verify Supabase connection

### AI Assistant Not Working
- Ensure `OPENAI_API_KEY` is set in Supabase Edge Functions
- Check edge function logs in Supabase Dashboard

### Build Errors
```bash
# Clear cache and reinstall
rm -rf node_modules pnpm-lock.yaml
pnpm install
pnpm run build
```

## 📝 Development

### Available Scripts

```bash
# Development server
pnpm run dev

# Type checking
pnpm run build

# Linting
pnpm run lint

# Preview production build
pnpm run preview
```

### Adding New Features

1. Update database schema in Supabase
2. Add types in `src/data/types.ts`
3. Create service functions in `src/lib/supabaseService.ts`
4. Build UI components in `src/components/`
5. Update pages in `src/pages/`

## 🌟 Future Enhancements

- [ ] Real-time race commentary
- [ ] Social features (comments, predictions sharing)
- [ ] Advanced analytics and visualizations
- [ ] Mobile app (React Native)
- [ ] Push notifications for race updates
- [ ] Multi-language support
- [ ] Historical data analysis tools

## 📄 License

This project is built for educational and demonstration purposes.

## 🙏 Acknowledgments

- **Ergast API**: F1 data provider
- **Supabase**: Backend infrastructure
- **Shadcn-UI**: UI component library
- **OpenAI**: AI assistant capabilities

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review Supabase logs
3. Check browser console for errors

---

Built with ❤️ for F1 fans by the MGX platform