/**
 * Drivers Page
 * Comprehensive driver profiles with analytics and performance metrics
 */
import { useState } from 'react';
import { useDrivers, useDriverStandings, useDriverById, useDriverResults } from '@/hooks/useBackendData';
import Header from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Search, Trophy, Loader2 } from 'lucide-react';
import { BackendStatus } from '@/components/BackendStatus';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import type { Driver, DriverStanding, RaceResult } from '@/types/api';

export default function DriversPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDriverId, setSelectedDriverId] = useState<string | null>(null);
  const currentYear = new Date().getFullYear();
  
  const { data: drivers, isLoading: driversLoading } = useDrivers(currentYear);
  const { data: standings, isLoading: standingsLoading } = useDriverStandings(currentYear);
  const { data: selectedDriver } = useDriverById(selectedDriverId || '');
  const { data: driverResults } = useDriverResults(selectedDriverId || '', currentYear);

  const filteredDrivers = drivers?.filter((driver: Driver) =>
    `${driver.given_name} ${driver.family_name}`.toLowerCase().includes(searchTerm.toLowerCase()) ||
    driver.code?.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  const getDriverStanding = (driverId: string): DriverStanding | undefined => {
    return standings?.find((s: DriverStanding) => s.driver_id === driverId);
  };

  const getInitials = (givenName: string, familyName: string) => {
    return `${givenName.charAt(0)}${familyName.charAt(0)}`;
  };

  const getPositionColor = (position: number) => {
    if (position === 1) return 'bg-yellow-500';
    if (position <= 3) return 'bg-gray-400';
    if (position <= 10) return 'bg-blue-500';
    return 'bg-gray-600';
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
              <h1 className="text-3xl font-bold tracking-tight">Drivers</h1>
              <p className="text-muted-foreground">Complete driver profiles and performance analytics</p>
            </div>
          </div>

          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search drivers by name or code..."
              className="w-full rounded-lg bg-background pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          {(driversLoading || standingsLoading) && (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </CardContent>
            </Card>
          )}

          {!driversLoading && !standingsLoading && (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
              {filteredDrivers.map((driver: Driver) => {
                const standing = getDriverStanding(driver.id);
                return (
                  <Card 
                    key={driver.id} 
                    className="cursor-pointer hover:shadow-lg transition-shadow"
                    onClick={() => setSelectedDriverId(driver.id)}
                  >
                    <CardHeader>
                      <div className="flex items-center gap-4">
                        <Avatar className="h-16 w-16">
                          <AvatarFallback className="text-lg font-bold">
                            {getInitials(driver.given_name, driver.family_name)}
                          </AvatarFallback>
                        </Avatar>
                        <div className="flex-1">
                          <CardTitle className="text-lg">
                            {driver.given_name} {driver.family_name}
                          </CardTitle>
                          <CardDescription>
                            {driver.code && <Badge variant="outline">{driver.code}</Badge>}
                          </CardDescription>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Position</span>
                          <div className="flex items-center gap-2">
                            <div className={`w-6 h-6 rounded-full ${getPositionColor(standing?.position || 99)} flex items-center justify-center`}>
                              <span className="text-xs font-bold text-white">
                                {standing?.position || '-'}
                              </span>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Points</span>
                          <span className="font-semibold">{standing?.points || 0}</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Wins</span>
                          <div className="flex items-center gap-1">
                            <Trophy className="h-3 w-3 text-yellow-600" />
                            <span className="font-semibold">{standing?.wins || 0}</span>
                          </div>
                        </div>
                        <div className="flex items-center justify-between">
                          <span className="text-sm text-muted-foreground">Nationality</span>
                          <span className="text-sm">{driver.nationality}</span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}

          {!driversLoading && filteredDrivers.length === 0 && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                <Search className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-semibold mb-2">No drivers found</p>
                <p className="text-sm text-muted-foreground">
                  Try adjusting your search criteria
                </p>
              </CardContent>
            </Card>
          )}

          <Dialog open={!!selectedDriverId} onOpenChange={(open) => !open && setSelectedDriverId(null)}>
            <DialogContent className="max-w-4xl max-h-[90vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>
                  {selectedDriver && `${selectedDriver.given_name} ${selectedDriver.family_name}`}
                </DialogTitle>
              </DialogHeader>
              
              {selectedDriver && (
                <div className="space-y-6">
                  <div className="grid gap-4 md:grid-cols-2">
                    <Card>
                      <CardHeader>
                        <CardTitle>Driver Information</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Code</span>
                          <span className="font-semibold">{selectedDriver.code || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Number</span>
                          <span className="font-semibold">{selectedDriver.permanent_number || 'N/A'}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Nationality</span>
                          <span className="font-semibold">{selectedDriver.nationality}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Date of Birth</span>
                          <span className="font-semibold">
                            {selectedDriver.date_of_birth ? 
                              new Date(selectedDriver.date_of_birth).toLocaleDateString() : 
                              'N/A'}
                          </span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Season Statistics</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        {getDriverStanding(selectedDriver.id) ? (
                          <>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Championship Position</span>
                              <span className="font-semibold">{getDriverStanding(selectedDriver.id)?.position}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Points</span>
                              <span className="font-semibold">{getDriverStanding(selectedDriver.id)?.points}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Wins</span>
                              <span className="font-semibold">{getDriverStanding(selectedDriver.id)?.wins}</span>
                            </div>
                          </>
                        ) : (
                          <p className="text-sm text-muted-foreground">No standings data available</p>
                        )}
                      </CardContent>
                    </Card>
                  </div>

                  {driverResults && driverResults.length > 0 && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Race Results - {currentYear}</CardTitle>
                        <CardDescription>Performance across the season</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                          <LineChart data={driverResults}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis 
                              dataKey="round" 
                              label={{ value: 'Race', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis 
                              reversed
                              domain={[1, 20]}
                              label={{ value: 'Position', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip />
                            <Legend />
                            <Line 
                              type="monotone" 
                              dataKey="position" 
                              stroke="#8884d8" 
                              strokeWidth={2}
                              name="Finish Position"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  )}

                  {driverResults && driverResults.length > 0 && (
                    <Card>
                      <CardHeader>
                        <CardTitle>Points Progress</CardTitle>
                        <CardDescription>Cumulative points throughout the season</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                          <BarChart data={driverResults}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="round" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="points" fill="#82ca9d" name="Points" />
                          </BarChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  )}
                </div>
              )}
            </DialogContent>
          </Dialog>
        </main>
      </div>
    </div>
  );
}