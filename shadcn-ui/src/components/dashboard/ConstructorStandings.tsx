import { useConstructorStandings } from '@/hooks/useBackendData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Trophy, Loader2, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import type { ConstructorStanding } from '@/types/api';

interface ConstructorStandingsProps {
  searchTerm?: string;
}

export default function ConstructorStandings({ searchTerm = '' }: ConstructorStandingsProps) {
  const { data: standings, isLoading, error } = useConstructorStandings();

  const filteredStandings = standings?.filter((standing: ConstructorStanding) =>
    standing.constructor_name?.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Constructor Standings</CardTitle>
          <CardDescription>Team championship positions</CardDescription>
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
          <CardTitle>Constructor Standings</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              Failed to load constructor standings. Please ensure the backend is running.
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
          <CardTitle>Constructor Standings</CardTitle>
          <CardDescription>Team championship positions</CardDescription>
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
        <CardTitle>Constructor Standings</CardTitle>
        <CardDescription>Team championship positions</CardDescription>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-12">Pos</TableHead>
              <TableHead>Constructor</TableHead>
              <TableHead className="text-right">Points</TableHead>
              <TableHead className="text-right">Wins</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredStandings.map((standing: ConstructorStanding) => (
              <TableRow key={standing.constructor_id}>
                <TableCell className="font-bold">
                  {standing.position === 1 && <Trophy className="h-4 w-4 inline text-yellow-600 mr-1" />}
                  {standing.position}
                </TableCell>
                <TableCell className="font-medium">{standing.constructor_name}</TableCell>
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