/**
 * Predictions Page
 * ML-powered race predictions with confidence scores and visualizations
 */
import { useState } from 'react';
import { useUpcomingRaces, usePredictions, useGeneratePredictions } from '@/hooks/useBackendData';
import Header from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Loader2, TrendingUp, Trophy, AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { BackendStatus } from '@/components/BackendStatus';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { Race, Prediction } from '@/types/api';

export default function PredictionsPage() {
  const [selectedRaceId, setSelectedRaceId] = useState<string>('');
  const { data: upcomingRaces, isLoading: racesLoading } = useUpcomingRaces();
  const { data: predictions, isLoading: predictionsLoading, error: predictionsError } = usePredictions(selectedRaceId);
  const generatePredictions = useGeneratePredictions();

  const handleGeneratePredictions = () => {
    if (selectedRaceId) {
      generatePredictions.mutate(selectedRaceId);
    }
  };

  const getConfidenceBadge = (confidence: number) => {
    if (confidence >= 0.7) return 'default';
    if (confidence >= 0.5) return 'secondary';
    return 'destructive';
  };

  return (
    <div className="grid min-h-screen w-full md:grid-cols-[220px_1fr] lg:grid-cols-[280px_1fr]">
      <Sidebar />
      <div className="flex flex-col">
        <Header />
        <main className="flex flex-1 flex-col gap-4 p-4 lg:gap-6 lg:p-6 bg-muted/40">
          <BackendStatus />
          
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold tracking-tight">Race Predictions</h1>
              <p className="text-muted-foreground">ML-powered predictions for upcoming races</p>
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Select Race</CardTitle>
              <CardDescription>Choose an upcoming race to view or generate predictions</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-4">
                <Select value={selectedRaceId} onValueChange={setSelectedRaceId}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select a race" />
                  </SelectTrigger>
                  <SelectContent>
                    {racesLoading ? (
                      <SelectItem value="loading" disabled>Loading races...</SelectItem>
                    ) : upcomingRaces && upcomingRaces.length > 0 ? (
                      upcomingRaces.map((race: Race) => (
                        <SelectItem key={race.id} value={race.id}>
                          {race.round}. {race.race_name} - {race.circuit_name}
                        </SelectItem>
                      ))
                    ) : (
                      <SelectItem value="none" disabled>No upcoming races</SelectItem>
                    )}
                  </SelectContent>
                </Select>
                <Button 
                  onClick={handleGeneratePredictions}
                  disabled={!selectedRaceId || generatePredictions.isPending}
                >
                  {generatePredictions.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Generating...
                    </>
                  ) : (
                    <>
                      <TrendingUp className="mr-2 h-4 w-4" />
                      Generate Predictions
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {predictionsLoading && (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </CardContent>
            </Card>
          )}

          {predictionsError && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>
                Failed to load predictions. Make sure the backend is running and try generating predictions.
              </AlertDescription>
            </Alert>
          )}

          {predictions && predictions.length > 0 && (
            <>
              <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                {predictions.slice(0, 3).map((prediction: Prediction, index: number) => (
                  <Card key={prediction.driver_id} className={index === 0 ? 'border-yellow-500 bg-yellow-50 dark:bg-yellow-950' : ''}>
                    <CardHeader>
                      <div className="flex items-center justify-between">
                        <CardTitle className="flex items-center gap-2">
                          {index === 0 && <Trophy className="h-5 w-5 text-yellow-600" />}
                          Position {index + 1}
                        </CardTitle>
                        <Badge variant={getConfidenceBadge(prediction.confidence)}>
                          {(prediction.confidence * 100).toFixed(0)}%
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <p className="text-2xl font-bold">{prediction.driver_name}</p>
                        <p className="text-sm text-muted-foreground">{prediction.constructor_name}</p>
                        <div className="pt-2">
                          <p className="text-xs text-muted-foreground">Confidence</p>
                          <div className="w-full bg-gray-200 rounded-full h-2 dark:bg-gray-700 mt-1">
                            <div 
                              className={`h-2 rounded-full ${
                                prediction.confidence >= 0.7 ? 'bg-green-600' :
                                prediction.confidence >= 0.5 ? 'bg-yellow-600' : 'bg-red-600'
                              }`}
                              style={{ width: `${prediction.confidence * 100}%` }}
                            />
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>Full Predictions</CardTitle>
                  <CardDescription>Complete race prediction with confidence scores</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {predictions.map((prediction: Prediction, index: number) => (
                      <div key={prediction.driver_id} className="flex items-center justify-between p-3 rounded-lg border">
                        <div className="flex items-center gap-4">
                          <span className="text-2xl font-bold text-muted-foreground w-8">{index + 1}</span>
                          <div>
                            <p className="font-semibold">{prediction.driver_name}</p>
                            <p className="text-sm text-muted-foreground">{prediction.constructor_name}</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <Badge variant={getConfidenceBadge(prediction.confidence)}>
                            {(prediction.confidence * 100).toFixed(1)}%
                          </Badge>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Confidence Distribution</CardTitle>
                  <CardDescription>Visual representation of prediction confidence levels</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={predictions.slice(0, 10)}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis 
                        dataKey="driver_name" 
                        angle={-45}
                        textAnchor="end"
                        height={100}
                      />
                      <YAxis />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="confidence" fill="#8884d8" name="Confidence" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </>
          )}

          {selectedRaceId && !predictionsLoading && (!predictions || predictions.length === 0) && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-semibold mb-2">No predictions available</p>
                <p className="text-sm text-muted-foreground mb-4">
                  Click "Generate Predictions" to create ML-powered race predictions
                </p>
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </div>
  );
}