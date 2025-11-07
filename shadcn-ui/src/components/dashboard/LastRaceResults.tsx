import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from '@/components/ui/pagination';
import { useQuery } from '@tanstack/react-query';
import { fetchLastRaceResults } from '@/lib/dataFetchingService';
import { useSortableData } from '@/hooks/useSortableData';
import { usePagination } from '@/hooks/usePagination';
import { Button } from '@/components/ui/button';
import { ArrowUpDown, Trophy } from 'lucide-react';
import LastRaceSkeleton from '../skeletons/LastRaceSkeleton';

export default function LastRaceResults() {
  const { data: race, isLoading } = useQuery({
    queryKey: ['lastRaceResults'],
    queryFn: fetchLastRaceResults,
  });

  const {
    items: sortedResults,
    requestSort,
  } = useSortableData(race?.Results || [], { key: 'position', direction: 'ascending' });

  const {
    currentData,
    currentPage,
    maxPage,
    next,
    prev,
  } = usePagination(sortedResults, 10);

  if (isLoading) {
    return <LastRaceSkeleton />;
  }

  if (!race) {
    return <div>No race results found.</div>;
  }

  const fastestLapDriver = race.Results.find(r => r.FastestLap?.rank === '1');

  return (
    <Card>
      <CardHeader>
        <CardTitle>Last Race Results: {race.raceName}</CardTitle>
        <CardDescription className="flex justify-between items-center">
          <span>{race.Circuit.circuitName} - {new Date(race.date).toLocaleDateString()}</span>
          {fastestLapDriver && (
            <span className="flex items-center gap-2 text-sm font-medium text-purple-600 dark:text-purple-400">
              <Trophy className="h-4 w-4" />
              Fastest Lap: {fastestLapDriver.Driver.givenName} {fastestLapDriver.Driver.familyName} ({fastestLapDriver.FastestLap?.Time.time})
            </span>
          )}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-auto h-[600px]">
          <Table>
            <TableHeader className="sticky-header">
              <TableRow>
                <TableHead>
                  <Button variant="ghost" onClick={() => requestSort('position')}>
                    Pos <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>Driver</TableHead>
                <TableHead>Constructor</TableHead>
                <TableHead>
                  <Button variant="ghost" onClick={() => requestSort('points')}>
                    Points <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {currentData.map((result) => (
                <TableRow key={result.Driver.driverId}>
                  <TableCell>{result.position}</TableCell>
                  <TableCell>
                    {result.Driver.givenName} {result.Driver.familyName}
                  </TableCell>
                  <TableCell>{result.Constructor.name}</TableCell>
                  <TableCell>{result.points}</TableCell>
                  <TableCell>{result.status}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        <Pagination className="mt-4">
          <PaginationContent>
            <PaginationItem>
              <PaginationPrevious href="#" onClick={() => prev()} />
            </PaginationItem>
            <PaginationItem>
              <PaginationLink href="#">{currentPage}</PaginationLink>
            </PaginationItem>
            <PaginationItem>
              <span className="px-2">/</span>
            </PaginationItem>
            <PaginationItem>
              <PaginationLink href="#">{maxPage}</PaginationLink>
            </PaginationItem>
            <PaginationItem>
              <PaginationNext href="#" onClick={() => next()} />
            </PaginationItem>
          </PaginationContent>
        </Pagination>
      </CardContent>
    </Card>
  );
}