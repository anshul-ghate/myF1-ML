# F1 Analytics Platform - Deployment Guide

This guide covers deploying the F1 Analytics Platform to production.

## Prerequisites

- Supabase project (already configured)
- OpenAI API key for AI assistant
- Domain name (optional)

## Deployment Steps

### 1. Environment Setup

The Supabase backend is already configured with:
- Database tables created
- Edge functions deployed
- Row Level Security enabled

### 2. Configure OpenAI API Key

1. Log in to [Supabase Dashboard](https://supabase.com/dashboard)
2. Navigate to your project: `hprhbsgmjjjgojkdasay`
3. Go to **Edge Functions** → **app_b64c9980ff_ai_assistant**
4. Click **Secrets** tab
5. Add new secret:
   - Key: `OPENAI_API_KEY`
   - Value: Your OpenAI API key

### 3. Initialize Backend Data

After deployment, you MUST initialize the database:

1. Navigate to `/admin` page
2. Click **"Sync F1 Data"** button
3. Wait for completion (30-60 seconds)
4. Click **"Generate Predictions"** button
5. Verify data appears on dashboard

### 4. Build and Deploy Frontend

#### Option A: Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
cd /workspace/shadcn-ui
vercel
```

#### Option B: Deploy to Netlify

```bash
# Install Netlify CLI
npm i -g netlify-cli

# Build
pnpm run build

# Deploy
netlify deploy --prod --dir=dist
```

#### Option C: Deploy to Supabase Storage (Static Hosting)

```bash
# Build
pnpm run build

# Upload to Supabase Storage
# (Use Supabase Dashboard → Storage → Upload dist folder)
```

### 5. Configure Authentication

1. In Supabase Dashboard → Authentication → URL Configuration
2. Add your production URL to:
   - Site URL
   - Redirect URLs

Example:
```
Site URL: https://your-domain.com
Redirect URLs: https://your-domain.com/**, http://localhost:5173/**
```

### 6. Set Up Custom Domain (Optional)

#### For Vercel:
1. Go to Project Settings → Domains
2. Add your custom domain
3. Configure DNS records as instructed

#### For Netlify:
1. Go to Site Settings → Domain Management
2. Add custom domain
3. Configure DNS records

### 7. Enable Email Authentication

1. Supabase Dashboard → Authentication → Providers
2. Enable Email provider
3. Configure email templates (optional)
4. Set up SMTP (optional, for custom emails)

## Post-Deployment Configuration

### Regular Maintenance Tasks

Set up automated tasks or manual reminders:

1. **After Each Race** (Sunday evening):
   - Run data sync from `/admin` page
   - Verify race results appear correctly

2. **Before Each Race** (Thursday):
   - Run prediction generation from `/admin` page
   - Verify predictions are available

3. **Weekly** (Optional):
   - Run data sync to catch any updates
   - Monitor edge function logs

### Monitoring

Monitor your deployment:

1. **Supabase Dashboard**:
   - Database → Check table row counts
   - Edge Functions → View logs and invocations
   - Authentication → Monitor user signups

2. **Application Logs**:
   - Browser console for frontend errors
   - Edge function logs for backend errors

### Performance Optimization

1. **Enable Caching**:
   - Configure CDN caching for static assets
   - Set appropriate cache headers

2. **Database Optimization**:
   - Indexes are already created
   - Monitor query performance in Supabase

3. **Edge Function Optimization**:
   - Functions are already optimized
   - Monitor cold start times

## Backup and Recovery

### Database Backup

Supabase provides automatic backups:
1. Dashboard → Database → Backups
2. Configure backup schedule (Pro plan)
3. Manual backups available anytime

### Restore Process

If you need to restore:
1. Supabase Dashboard → Database → Backups
2. Select backup point
3. Click "Restore"

## Security Checklist

- [x] RLS policies enabled on all tables
- [x] Environment variables secured
- [x] HTTPS enabled (automatic with Vercel/Netlify)
- [ ] OpenAI API key added to edge functions
- [ ] Production URLs added to Supabase auth config
- [ ] Rate limiting configured (optional)

## Troubleshooting

### Issue: No data showing after deployment

**Solution**:
1. Visit `/admin` page
2. Run "Sync F1 Data"
3. Check edge function logs for errors

### Issue: AI Assistant not responding

**Solution**:
1. Verify `OPENAI_API_KEY` is set in Supabase
2. Check edge function logs
3. Verify OpenAI API key is valid

### Issue: Authentication not working

**Solution**:
1. Check Site URL and Redirect URLs in Supabase
2. Verify email provider is enabled
3. Check browser console for errors

### Issue: Build fails

**Solution**:
```bash
# Clear cache
rm -rf node_modules pnpm-lock.yaml dist

# Reinstall
pnpm install

# Rebuild
pnpm run build
```

## Scaling Considerations

### Database Scaling

Supabase automatically scales, but monitor:
- Connection pool usage
- Query performance
- Storage usage

### Edge Function Scaling

Edge functions auto-scale, but consider:
- Cold start times
- Concurrent invocation limits
- Timeout settings (currently 10s)

### Frontend Scaling

Static hosting (Vercel/Netlify) scales automatically:
- Global CDN distribution
- Automatic SSL
- DDoS protection

## Cost Estimation

### Supabase Costs

**Free Tier Includes**:
- 500MB database
- 2GB bandwidth
- 500K edge function invocations
- 50K monthly active users

**Estimated Usage**:
- Database: ~50MB (F1 data)
- Edge Functions: ~1K invocations/day
- Bandwidth: ~100MB/day

**Recommendation**: Free tier sufficient for initial launch

### OpenAI Costs

**GPT-4o-mini Pricing**:
- Input: $0.150 / 1M tokens
- Output: $0.600 / 1M tokens

**Estimated Usage**:
- ~100 queries/day
- ~500 tokens per query
- Cost: ~$1-2/day

## Monitoring and Analytics

### Recommended Tools

1. **Supabase Dashboard**: Built-in analytics
2. **Google Analytics**: User behavior tracking
3. **Sentry**: Error tracking (optional)
4. **Vercel/Netlify Analytics**: Performance metrics

### Key Metrics to Track

- Daily active users
- API response times
- Edge function invocations
- Database query performance
- Error rates

## Support and Maintenance

### Regular Updates

1. **Weekly**: Check for Supabase updates
2. **After races**: Update F1 data
3. **Monthly**: Review and optimize performance
4. **Quarterly**: Security audit

### Getting Help

- Supabase Discord: https://discord.supabase.com
- Documentation: https://supabase.com/docs
- GitHub Issues: For bug reports

---

**Deployment Checklist**:
- [ ] OpenAI API key configured
- [ ] Backend data initialized
- [ ] Frontend deployed
- [ ] Custom domain configured (optional)
- [ ] Authentication URLs updated
- [ ] Monitoring set up
- [ ] Backup schedule confirmed
- [ ] Security checklist completed

Congratulations! Your F1 Analytics Platform is now live! 🏎️🎉