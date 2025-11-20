import { useDriverStandings } from '@/hooks/useBackendData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Trophy, Loader2, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import type { DriverStanding } from '@/types/api';

interface DriverStandingsProps {
  searchTerm?: string;
}

export default function DriverStandings({ searchTerm = '' }: DriverStandingsProps) {
  const { data: standings, isLoading, error } = useDriverStandings();

  const filteredStandings = standings?.filter((standing: DriverStanding) =>
    standing.driver_name?.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Driver Standings</CardTitle>
          <CardDescription>Current championship positions</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Driver Standings</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Failed to load driver standings. Please ensure the backend is running.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!standings || standings.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Driver Standings</CardTitle>
          <CardDescription>Current championship positions</CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-center text-muted-foreground py-8">No standings data available</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Driver Standings</CardTitle>
        <CardDescription>Current championship positions</CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">Pos</TableHead>
              <TableHead>Driver</TableHead>
              <TableHead>Team</TableHead>
              <TableHead className="text-right">Points</TableHead>
              <TableHead className="text-right">Wins</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredStandings.map((standing: DriverStanding) => (
              <TableRow key={standing.driver_id}>
                <TableCell className="font-bold">
                  {standing.position === 1 && <Trophy className="h-4 w-4 inline text-yellow-600 mr-1" />}
                  {standing.position}
                </TableCell>
                <TableCell className="font-medium">{standing.driver_name}</TableCell>
                <TableCell>{standing.constructor_name}</TableCell>
                <TableCell className="text-right">
                  <Badge variant="secondary">{standing.points}</Badge>
                </TableCell>
                <TableCell className="text-right">{standing.wins}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}