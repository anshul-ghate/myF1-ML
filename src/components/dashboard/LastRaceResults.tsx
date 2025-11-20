import { useRaces } from '@/hooks/useBackendData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Trophy, Loader2, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import type { Race, RaceResult } from '@/types/api';

export default function LastRaceResults() {
  const currentYear = new Date().getFullYear();
  const { data: races, isLoading, error } = useRaces(currentYear);

  // Get the most recent completed race
  const lastRace = races?.find((race: Race) => 
    new Date(race.date) < new Date() && race.results && race.results.length > 0
  );

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Last Race Results</CardTitle>
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
          <CardTitle>Last Race Results</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Failed to load race results. Please ensure the backend is running.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!lastRace) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Last Race Results</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-center text-muted-foreground py-8">No recent race results available</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Last Race Results</CardTitle>
        <CardDescription>
          {lastRace.race_name} - {lastRace.circuit_name}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">Pos</TableHead>
              <TableHead>Driver</TableHead>
              <TableHead>Team</TableHead>
              <TableHead className="text-right">Points</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {lastRace.results?.slice(0, 10).map((result: RaceResult) => (
              <TableRow key={result.driver_id}>
                <TableCell className="font-bold">
                  {result.position === 1 && <Trophy className="h-4 w-4 inline text-yellow-600 mr-1" />}
                  {result.position}
                </TableCell>
                <TableCell className="font-medium">{result.driver_name}</TableCell>
                <TableCell>{result.constructor_name}</TableCell>
                <TableCell className="text-right">
                  <Badge variant="secondary">{result.points}</Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
}