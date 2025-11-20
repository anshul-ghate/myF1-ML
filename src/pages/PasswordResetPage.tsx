import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { supabase } from '@/lib/supabaseClient';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';

export default function PasswordResetPage() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [accessToken, setAccessToken] = useState<string | null>(null);

  useEffect(() => {
    const hash = window.location.hash;
    const params = new URLSearchParams(hash.substring(1)); // remove #
    const token = params.get('access_token');
    if (token) {
      setAccessToken(token);
    } else {
        // Handle cases where the fragment is not present or doesn't contain the token
        const url = new URL(window.location.href);
        const error_description = url.searchParams.get('error_description');
        if(error_description) {
            toast({
                title: 'Error',
                description: error_description,
                variant: 'destructive',
            });
        }
    }
  }, [toast]);

  const handlePasswordReset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!accessToken) {
        toast({
            title: 'Error',
            description: 'Invalid or missing reset token.',
            variant: 'destructive',
        });
        return;
    }

    setLoading(true);
    const { error } = await supabase.auth.updateUser({ password });

    if (error) {
      toast({
        title: 'Error resetting password',
        description: error.message,
        variant: 'destructive',
      });
    } else {
      toast({
        title: 'Password reset successful',
        description: 'You can now log in with your new password.',
      });
      navigate('/login');
    }
    setLoading(false);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Reset Password</CardTitle>
          <CardDescription>Enter your new password below.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handlePasswordReset}>
            <div className="grid gap-4">
              <div className="grid gap-2">
                <Label htmlFor="password">New Password</Label>
                <Input
                  id="password"
                  type="password"
                  required
                  minLength={6}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
              <Button type="submit" className="w-full" disabled={loading || !accessToken}>
                {loading ? 'Resetting...' : 'Reset Password'}
              </Button>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}