import { useEffect, useState } from 'react';
import { useUpcomingRaces } from '@/hooks/useBackendData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Calendar, MapPin, Clock, Loader2 } from 'lucide-react';

export function RaceCountdown() {
  const { data: upcomingRaces, isLoading } = useUpcomingRaces();
  const [timeLeft, setTimeLeft] = useState<string>('');

  const nextRace = upcomingRaces?.[0];

  useEffect(() => {
    if (!nextRace?.date) return;

    const calculateTimeLeft = () => {
      const raceDate = new Date(nextRace.date);
      const now = new Date();
      const difference = raceDate.getTime() - now.getTime();

      if (difference > 0) {
        const days = Math.floor(difference / (1000 * 60 * 60 * 24));
        const hours = Math.floor((difference / (1000 * 60 * 60)) % 24);
        const minutes = Math.floor((difference / 1000 / 60) % 60);
        const seconds = Math.floor((difference / 1000) % 60);

        setTimeLeft(`${days}d ${hours}h ${minutes}m ${seconds}s`);
      } else {
        setTimeLeft('Race in progress or completed');
      }
    };

    calculateTimeLeft();
    const timer = setInterval(calculateTimeLeft, 1000);

    return () => clearInterval(timer);
  }, [nextRace]);

  if (isLoading) {
    return (
      <Card className="bg-gradient-to-br from-red-500 to-red-700 text-white">
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin" />
        </CardContent>
      </Card>
    );
  }

  if (!nextRace) {
    return (
      <Card className="bg-gradient-to-br from-gray-500 to-gray-700 text-white">
        <CardHeader>
          <CardTitle>Next Race</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center">No upcoming races scheduled</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="bg-gradient-to-br from-red-500 to-red-700 text-white">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-2xl">{nextRace.race_name}</CardTitle>
            <CardDescription className="text-red-100">
              Round {nextRace.round} of {nextRace.season}
            </CardDescription>
          </div>
          <Badge variant="secondary" className="bg-white text-red-700">
            Next Race
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-center gap-2">
            <MapPin className="h-5 w-5" />
            <div>
              <p className="text-sm opacity-90">Circuit</p>
              <p className="font-semibold">{nextRace.circuit_name}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Calendar className="h-5 w-5" />
            <div>
              <p className="text-sm opacity-90">Date</p>
              <p className="font-semibold">
                {new Date(nextRace.date).toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric',
                })}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            <div>
              <p className="text-sm opacity-90">Countdown</p>
              <p className="font-semibold font-mono">{timeLeft}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}