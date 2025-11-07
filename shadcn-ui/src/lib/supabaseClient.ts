import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://hprhbsgmjjjgojkdasay.supabase.co';
const supabaseAnonKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imhwcmhic2dtampqZ29qa2Rhc2F5Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjI0NjQxMTAsImV4cCI6MjA3ODA0MDExMH0.MV3kUYADVheoCPyGW8Axcj_hDIOVa6n0Zn-nfs7ckuY';

export const supabase = createClient(supabaseUrl, supabaseAnonKey);