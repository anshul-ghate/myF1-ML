import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { useQuery } from '@tanstack/react-query';
import { fetchRaceSchedule, fetchNextRace } from '@/lib/dataFetchingService';
import { Race } from '@/data/types';
import { cn } from '@/lib/utils';

export default function RaceSchedule() {
  const { data: schedule, isLoading: isLoadingSchedule } = useQuery({
    queryKey: ['raceSchedule'],
    queryFn: fetchRaceSchedule,
  });

  const { data: nextRace, isLoading: isLoadingNextRace } = useQuery({
    queryKey: ['nextRace'],
    queryFn: fetchNextRace,
  });

  if (isLoadingSchedule || isLoadingNextRace) {
    return <div>Loading Race Schedule...</div>;
  }

  const formatRaceTime = (date: string, time: string) => {
    const dateTimeString = `${date}T${time}`;
    return new Date(dateTimeString).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="max-h-[70vh] overflow-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Round</TableHead>
            <TableHead>Race Name</TableHead>
            <TableHead>Circuit</TableHead>
            <TableHead>Date</TableHead>
            <TableHead>Time</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {schedule?.map((race: Race) => (
            <TableRow
              key={race.round}
              className={cn(
                nextRace && race.round === nextRace.round
                  ? 'bg-primary/10'
                  : ''
              )}
            >
              <TableCell>{race.round}</TableCell>
              <TableCell>{race.raceName}</TableCell>
              <TableCell>{race.Circuit.circuitName}</TableCell>
              <TableCell>{new Date(race.date).toLocaleDateString()}</TableCell>
              <TableCell>{race.time ? formatRaceTime(race.date, race.time) : 'TBA'}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}