/**
 * Strategy Simulator Page
 * Interactive pit stop strategy builder with Monte Carlo simulations
 */
import { useState } from 'react';
import { useUpcomingRaces, useDrivers, useSimulateStrategy, useOptimizeStrategies } from '@/hooks/useBackendData';
import Header from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Loader2, Plus, Trash2, Zap } from 'lucide-react';
import { BackendStatus } from '@/components/BackendStatus';
import type { Race, Driver, PitStop, StrategySimulation } from '@/types/api';

interface LocalPitStop {
  lap: number;
  compound: string;
}

export default function StrategySimulatorPage() {
  const [selectedRaceId, setSelectedRaceId] = useState<string>('');
  const [selectedDriverId, setSelectedDriverId] = useState<string>('');
  const [pitStops, setPitStops] = useState<LocalPitStop[]>([{ lap: 20, compound: 'MEDIUM' }]);
  
  const { data: upcomingRaces, isLoading: racesLoading } = useUpcomingRaces();
  const { data: drivers, isLoading: driversLoading } = useDrivers();
  const simulateStrategy = useSimulateStrategy();
  const optimizeStrategies = useOptimizeStrategies();

  const addPitStop = () => {
    setPitStops([...pitStops, { lap: 40, compound: 'SOFT' }]);
  };

  const removePitStop = (index: number) => {
    setPitStops(pitStops.filter((_, i) => i !== index));
  };

  const updatePitStop = (index: number, field: keyof LocalPitStop, value: string | number) => {
    const updated = [...pitStops];
    updated[index] = { ...updated[index], [field]: value };
    setPitStops(updated);
  };

  const handleSimulate = () => {
    if (selectedRaceId && selectedDriverId) {
      simulateStrategy.mutate({
        race_id: selectedRaceId,
        driver_id: selectedDriverId,
        pit_stops: pitStops as PitStop[],
      });
    }
  };

  const handleOptimize = () => {
    if (selectedRaceId && selectedDriverId) {
      optimizeStrategies.mutate({
        race_id: selectedRaceId,
        driver_id: selectedDriverId,
        num_strategies: 5,
      });
    }
  };

  const getTireColor = (compound: string) => {
    switch (compound.toUpperCase()) {
      case 'SOFT': return 'bg-red-500';
      case 'MEDIUM': return 'bg-yellow-500';
      case 'HARD': return 'bg-white border border-gray-300';
      case 'INTERMEDIATE': return 'bg-green-500';
      case 'WET': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
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
              <h1 className="text-3xl font-bold tracking-tight">Strategy Simulator</h1>
              <p className="text-muted-foreground">Monte Carlo pit stop strategy optimization</p>
            </div>
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Race Configuration</CardTitle>
                <CardDescription>Select race and driver for simulation</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Race</Label>
                  <Select value={selectedRaceId} onValueChange={setSelectedRaceId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a race" />
                    </SelectTrigger>
                    <SelectContent>
                      {racesLoading ? (
                        <SelectItem value="loading" disabled>Loading...</SelectItem>
                      ) : upcomingRaces && upcomingRaces.length > 0 ? (
                        upcomingRaces.map((race: Race) => (
                          <SelectItem key={race.id} value={race.id}>
                            {race.race_name}
                          </SelectItem>
                        ))
                      ) : (
                        <SelectItem value="none" disabled>No races</SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label>Driver</Label>
                  <Select value={selectedDriverId} onValueChange={setSelectedDriverId}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a driver" />
                    </SelectTrigger>
                    <SelectContent>
                      {driversLoading ? (
                        <SelectItem value="loading" disabled>Loading...</SelectItem>
                      ) : drivers && drivers.length > 0 ? (
                        drivers.map((driver: Driver) => (
                          <SelectItem key={driver.id} value={driver.id}>
                            {driver.given_name} {driver.family_name}
                          </SelectItem>
                        ))
                      ) : (
                        <SelectItem value="none" disabled>No drivers</SelectItem>
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Pit Stop Strategy</CardTitle>
                <CardDescription>Configure pit stops and tire compounds</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {pitStops.map((stop, index) => (
                  <div key={index} className="flex gap-2 items-end">
                    <div className="flex-1 space-y-2">
                      <Label>Lap {index + 1}</Label>
                      <Input
                        type="number"
                        value={stop.lap}
                        onChange={(e) => updatePitStop(index, 'lap', parseInt(e.target.value))}
                        min={1}
                        max={70}
                      />
                    </div>
                    <div className="flex-1 space-y-2">
                      <Label>Compound</Label>
                      <Select 
                        value={stop.compound} 
                        onValueChange={(value) => updatePitStop(index, 'compound', value)}
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="SOFT">
                            <div className="flex items-center gap-2">
                              <div className={`w-3 h-3 rounded-full ${getTireColor('SOFT')}`} />
                              Soft
                            </div>
                          </SelectItem>
                          <SelectItem value="MEDIUM">
                            <div className="flex items-center gap-2">
                              <div className={`w-3 h-3 rounded-full ${getTireColor('MEDIUM')}`} />
                              Medium
                            </div>
                          </SelectItem>
                          <SelectItem value="HARD">
                            <div className="flex items-center gap-2">
                              <div className={`w-3 h-3 rounded-full ${getTireColor('HARD')}`} />
                              Hard
                            </div>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Button
                      variant="destructive"
                      size="icon"
                      onClick={() => removePitStop(index)}
                      disabled={pitStops.length === 1}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
                <Button onClick={addPitStop} variant="outline" className="w-full">
                  <Plus className="mr-2 h-4 w-4" />
                  Add Pit Stop
                </Button>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardContent className="pt-6">
              <div className="flex gap-4">
                <Button 
                  onClick={handleSimulate}
                  disabled={!selectedRaceId || !selectedDriverId || simulateStrategy.isPending}
                  className="flex-1"
                >
                  {simulateStrategy.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Simulating...
                    </>
                  ) : (
                    'Simulate Strategy'
                  )}
                </Button>
                <Button 
                  onClick={handleOptimize}
                  disabled={!selectedRaceId || !selectedDriverId || optimizeStrategies.isPending}
                  variant="secondary"
                  className="flex-1"
                >
                  {optimizeStrategies.isPending ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Optimizing...
                    </>
                  ) : (
                    <>
                      <Zap className="mr-2 h-4 w-4" />
                      Auto-Optimize
                    </>
                  )}
                </Button>
              </div>
            </CardContent>
          </Card>

          {simulateStrategy.data && (
            <div className="grid gap-4 md:grid-cols-3">
              <Card>
                <CardHeader>
                  <CardTitle>Predicted Time</CardTitle>
                  <CardDescription>Total race time estimate</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-bold">
                    {simulateStrategy.data.mean_time ? 
                      `${Math.floor(simulateStrategy.data.mean_time / 60)}:${(simulateStrategy.data.mean_time % 60).toFixed(3)}` 
                      : 'N/A'}
                  </p>
                  <p className="text-sm text-muted-foreground mt-2">
                    ±{simulateStrategy.data.std_dev?.toFixed(2)}s variance
                  </p>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Risk Score</CardTitle>
                  <CardDescription>Strategy risk assessment</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-2">
                    <p className="text-3xl font-bold">
                      {simulateStrategy.data.risk_score ? 
                        (simulateStrategy.data.risk_score * 100).toFixed(0) 
                        : 'N/A'}
                    </p>
                    <Badge variant={
                      simulateStrategy.data.risk_score > 0.7 ? 'destructive' :
                      simulateStrategy.data.risk_score > 0.4 ? 'secondary' : 'default'
                    }>
                      {simulateStrategy.data.risk_score > 0.7 ? 'High' :
                       simulateStrategy.data.risk_score > 0.4 ? 'Medium' : 'Low'}
                    </Badge>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Simulations</CardTitle>
                  <CardDescription>Monte Carlo iterations</CardDescription>
                </CardHeader>
                <CardContent>
                  <p className="text-3xl font-bold">1,000+</p>
                  <p className="text-sm text-muted-foreground mt-2">
                    Completed successfully
                  </p>
                </CardContent>
              </Card>
            </div>
          )}

          {optimizeStrategies.data && optimizeStrategies.data.length > 0 && (
            <Card>
              <CardHeader>
                <CardTitle>Optimized Strategies</CardTitle>
                <CardDescription>Top 5 strategies ranked by predicted performance</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {optimizeStrategies.data.map((strategy: StrategySimulation, index: number) => (
                    <div key={index} className="p-4 border rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-semibold">Strategy {index + 1}</h3>
                        <Badge>{strategy.mean_time ? `${Math.floor(strategy.mean_time / 60)}:${(strategy.mean_time % 60).toFixed(2)}` : 'N/A'}</Badge>
                      </div>
                      <div className="flex gap-2 flex-wrap">
                        {strategy.pit_stops?.map((stop: PitStop, stopIndex: number) => (
                          <div key={stopIndex} className="flex items-center gap-1 text-sm">
                            <span>Lap {stop.lap}:</span>
                            <div className={`w-3 h-3 rounded-full ${getTireColor(stop.compound)}`} />
                            <span>{stop.compound}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          )}
        </main>
      </div>
    </div>
  );
}