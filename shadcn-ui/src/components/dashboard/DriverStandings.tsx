import { useMemo, useState } from 'react';
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
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  Pagination,
  PaginationContent,
  PaginationItem,
  PaginationLink,
  PaginationNext,
  PaginationPrevious,
} from '@/components/ui/pagination';
import { useQuery } from '@tanstack/react-query';
import { DataFetchingService } from '@/lib/dataFetchingService';
import { useSortableData } from '@/hooks/useSortableData';
import { usePagination } from '@/hooks/usePagination';
import { Button } from '@/components/ui/button';
import { ArrowUpDown } from 'lucide-react';
import type { DriverStanding } from '@/data/types';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import DriverDetails from '../details/DriverDetails';
import StandingsSkeleton from '../skeletons/StandingsSkeleton';

interface DriverStandingsProps {
  searchTerm: string;
}

export default function DriverStandings({ searchTerm }: DriverStandingsProps) {
  const { data: standings, isLoading } = useQuery({
    queryKey: ['driverStandings'],
    queryFn: DataFetchingService.getDriverStandings,
  });
  const [selectedStanding, setSelectedStanding] = useState<DriverStanding | null>(null);

  const filteredStandings = useMemo(() => {
    if (!standings) return [];
    return standings.filter(
      (s) =>
        s.Driver.givenName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.Driver.familyName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        s.Constructors[0].name.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [standings, searchTerm]);

  const {
    items: sortedStandings,
    requestSort,
  } = useSortableData(filteredStandings, { key: 'position', direction: 'ascending' });

  const {
    currentData,
    currentPage,
    maxPage,
    next,
    prev,
  } = usePagination(sortedStandings, 10);

  if (isLoading) {
    return <StandingsSkeleton />;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Driver Standings</CardTitle>
        <CardDescription>Current 2024 Season Standings</CardDescription>
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
              </TableRow>
            </TableHeader>
            <TableBody>
              {currentData().map((standing: DriverStanding) => (
                <Dialog key={standing.Driver.driverId} onOpenChange={(isOpen) => !isOpen && setSelectedStanding(null)}>
                  <DialogTrigger asChild>
                    <TableRow onClick={() => setSelectedStanding(standing)} className="cursor-pointer">
                      <TableCell>{standing.position}</TableCell>
                      <TableCell className="flex items-center gap-2">
                        <Avatar className="h-8 w-8">
                          <AvatarImage src="/assets/driver-placeholder.png" alt={standing.Driver.givenName} />
                          <AvatarFallback>{standing.Driver.code}</AvatarFallback>
                        </Avatar>
                        {standing.Driver.givenName} {standing.Driver.familyName}
                      </TableCell>
                      <TableCell>{standing.Constructors[0].name}</TableCell>
                      <TableCell>{standing.points}</TableCell>
                    </TableRow>
                  </DialogTrigger>
                  {selectedStanding && selectedStanding.Driver.driverId === standing.Driver.driverId && (
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Driver Details</DialogTitle>
                      </DialogHeader>
                      <DriverDetails driver={selectedStanding.Driver} standing={selectedStanding} />
                    </DialogContent>
                  )}
                </Dialog>
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