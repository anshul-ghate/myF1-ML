import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { DataFetchingService } from '@/lib/dataFetchingService';
import type { Race } from '@/data/types';

interface TimeLeft {
  days?: number;
  hours?: number;
  minutes?: number;
  seconds?: number;
}

const calculateTimeLeft = (raceDate: string): TimeLeft => {
  const difference = +new Date(raceDate) - +new Date();
  let timeLeft: TimeLeft = {};

  if (difference > 0) {
    timeLeft = {
      days: Math.floor(difference / (1000 * 60 * 60 * 24)),
      hours: Math.floor((difference / (1000 * 60 * 60)) % 24),
      minutes: Math.floor((difference / 1000 / 60) % 60),
      seconds: Math.floor((difference / 1000) % 60),
    };
  }
  return timeLeft;
};

export function RaceCountdown() {
  const [nextRace, setNextRace] = useState<Race | null>(null);
  const [timeLeft, setTimeLeft] = useState<TimeLeft>({});

  useEffect(() => {
    DataFetchingService.getNextRace().then((race) => {
      setNextRace(race);
      setTimeLeft(calculateTimeLeft(race.date));
    });
  }, []);

  useEffect(() => {
    if (!nextRace) return;
    const timer = setTimeout(() => {
      setTimeLeft(calculateTimeLeft(nextRace.date));
    }, 1000);
    return () => clearTimeout(timer);
  });

  if (!nextRace) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Next Race</CardTitle>
        </CardHeader>
        <CardContent>
          <p>Loading next race...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Next Race: {nextRace.raceName}</CardTitle>
        <p className="text-sm text-muted-foreground">{nextRace.Circuit.circuitName} - {new Date(nextRace.date).toLocaleDateString()}</p>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-4 gap-4 text-center">
          <div>
            <p className="text-2xl font-bold">{timeLeft.days || '0'}</p>
            <p className="text-sm text-muted-foreground">Days</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{timeLeft.hours || '0'}</p>
            <p className="text-sm text-muted-foreground">Hours</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{timeLeft.minutes || '0'}</p>
            <p className="text-sm text-muted-foreground">Minutes</p>
          </div>
          <div>
            <p className="text-2xl font-bold">{timeLeft.seconds || '0'}</p>
            <p className="text-sm text-muted-foreground">Seconds</p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}