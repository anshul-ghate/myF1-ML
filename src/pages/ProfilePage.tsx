import { useState, useEffect } from 'react';
import { supabase } from '@/lib/supabaseClient';
import { useAuth } from '@/contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

export default function ProfilePage() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [loading, setLoading] = useState(true);
  const [fullName, setFullName] = useState('');
  const [website, setWebsite] = useState('');
  const [avatarUrl, setAvatarUrl] = useState('');

  useEffect(() => {
    let ignore = false;
    async function getProfile() {
      setLoading(true);
      if (user) {
        const { data, error } = await supabase
          .from('app_b64c9980ff_profiles')
          .select(`full_name, website, avatar_url`)
          .eq('id', user.id)
          .single();

        if (!ignore) {
          if (error) {
            console.warn(error);
            toast({ title: "Error fetching profile", description: error.message, variant: "destructive" });
          } else if (data) {
            setFullName(data.full_name || '');
            setWebsite(data.website || '');
            setAvatarUrl(data.avatar_url || '');
          }
        }
      }
      setLoading(false);
    }

    getProfile();

    return () => {
      ignore = true;
    };
  }, [user, toast]);

  async function updateProfile(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    if (user) {
      const updates = {
        id: user.id,
        full_name: fullName,
        website,
        avatar_url: avatarUrl,
        updated_at: new Date(),
      };

      const { error } = await supabase.from('app_b64c9980ff_profiles').upsert(updates);

      if (error) {
        toast({ title: "Error updating profile", description: error.message, variant: "destructive" });
      } else {
        toast({ title: "Profile updated successfully!" });
      }
    }
    setLoading(false);
  }
  
  async function handleSignOut() {
    setLoading(true);
    const { error } = await supabase.auth.signOut();
    if (error) {
        toast({ title: "Error signing out", description: error.message, variant: "destructive" });
    }
    setLoading(false);
  }

  return (
    <div className="flex justify-center items-center min-h-screen bg-gray-50">
      <Card className="w-full max-w-lg">
        <CardHeader>
          <CardTitle>User Profile</CardTitle>
          <CardDescription>Update your profile information.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={updateProfile} className="space-y-6">
            <div className="flex items-center space-x-4">
              <Avatar className="h-24 w-24">
                <AvatarImage src={avatarUrl || `https://api.dicebear.com/6.x/initials/svg?seed=${user?.email}`} alt="Avatar" />
                <AvatarFallback>{fullName?.charAt(0) || user?.email?.charAt(0)}</AvatarFallback>
              </Avatar>
              {/* Add avatar upload functionality here in the future */}
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input id="email" type="email" value={user?.email || ''} disabled />
            </div>
            <div className="space-y-2">
              <Label htmlFor="fullName">Full Name</Label>
              <Input
                id="fullName"
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="website">Website</Label>
              <Input
                id="website"
                type="url"
                value={website}
                onChange={(e) => setWebsite(e.target.value)}
              />
            </div>

            <div className="flex justify-between">
              <Button type="submit" disabled={loading}>
                {loading ? 'Updating...' : 'Update Profile'}
              </Button>
              <Button variant="outline" onClick={handleSignOut} disabled={loading}>
                Sign Out
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}