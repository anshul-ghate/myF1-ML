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
import { fetchConstructorStandings } from '@/lib/dataFetchingService';
import { useSortableData } from '@/hooks/useSortableData';
import { usePagination } from '@/hooks/usePagination';
import { Button } from '@/components/ui/button';
import { ArrowUpDown, AlertCircle } from 'lucide-react';
import type { ConstructorStanding } from '@/data/types';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import ConstructorDetails from '../details/ConstructorDetails';
import StandingsSkeleton from '../skeletons/StandingsSkeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';

interface ConstructorStandingsProps {
  searchTerm: string;
}

export default function ConstructorStandings({ searchTerm }: ConstructorStandingsProps) {
  const { data: standings, isLoading } = useQuery({
    queryKey: ['constructorStandings'],
    queryFn: fetchConstructorStandings,
  });
  const [selectedStanding, setSelectedStanding] = useState<ConstructorStanding | null>(null);

  const filteredStandings = useMemo(() => {
    if (!standings || !Array.isArray(standings)) return [];
    return standings.filter((s) =>
      s?.Constructor?.name?.toLowerCase().includes(searchTerm.toLowerCase())
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
  } = usePagination(sortedStandings, 5);

  if (isLoading) {
    return <StandingsSkeleton />;
  }

  if (!standings || standings.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Constructor Standings</CardTitle>
          <CardDescription>Current 2024 Season Standings</CardDescription>
        </CardHeader>
        <CardContent>
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>No Data Available</AlertTitle>
            <AlertDescription>
              Please initialize the backend by visiting the <a href="/admin" className="underline font-semibold">Admin page</a> and clicking "Sync F1 Data".
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Constructor Standings</CardTitle>
        <CardDescription>Current 2024 Season Standings</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="overflow-auto h-[380px]">
          <Table>
            <TableHeader className="sticky-header">
              <TableRow>
                <TableHead>
                  <Button variant="ghost" onClick={() => requestSort('position')}>
                    Pos <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
                <TableHead>Constructor</TableHead>
                <TableHead>
                  <Button variant="ghost" onClick={() => requestSort('points')}>
                    Points <ArrowUpDown className="ml-2 h-4 w-4" />
                  </Button>
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {currentData.map((standing: ConstructorStanding) => (
                <Dialog key={standing.Constructor.constructorId} onOpenChange={(isOpen) => !isOpen && setSelectedStanding(null)}>
                  <DialogTrigger asChild>
                    <TableRow onClick={() => setSelectedStanding(standing)} className="cursor-pointer">
                      <TableCell>{standing.position}</TableCell>
                      <TableCell className="flex items-center gap-2">
                        <Avatar className="h-8 w-8">
                          <AvatarImage src="/assets/constructor-placeholder.png" alt={standing.Constructor.name} />
                          <AvatarFallback>{standing.Constructor.name.substring(0, 2)}</AvatarFallback>
                        </Avatar>
                        {standing.Constructor.name}
                      </TableCell>
                      <TableCell>{standing.points}</TableCell>
                    </TableRow>
                  </DialogTrigger>
                  {selectedStanding && selectedStanding.Constructor.constructorId === standing.Constructor.constructorId && (
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Constructor Details</DialogTitle>
                      </DialogHeader>
                      <ConstructorDetails constructor={selectedStanding.Constructor} standing={selectedStanding} />
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