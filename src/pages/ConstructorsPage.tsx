/**
 * Constructors Page
 * Team statistics and performance metrics
 */
import { useState } from 'react';
import { useConstructors, useConstructorStandings, useConstructorById } from '@/hooks/useBackendData';
import Header from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Search, Trophy, Loader2, Building2 } from 'lucide-react';
import { BackendStatus } from '@/components/BackendStatus';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import type { Constructor, ConstructorStanding } from '@/types/api';

export default function ConstructorsPage() {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedConstructorId, setSelectedConstructorId] = useState<string | null>(null);
  const currentYear = new Date().getFullYear();
  
  const { data: constructors, isLoading: constructorsLoading } = useConstructors(currentYear);
  const { data: standings, isLoading: standingsLoading } = useConstructorStandings(currentYear);
  const { data: selectedConstructor } = useConstructorById(selectedConstructorId || '');

  const filteredConstructors = constructors?.filter((constructor: Constructor) =>
    constructor.name.toLowerCase().includes(searchTerm.toLowerCase())
  ) || [];

  const getConstructorStanding = (constructorId: string): ConstructorStanding | undefined => {
    return standings?.find((s: ConstructorStanding) => s.constructor_id === constructorId);
  };

  const getPositionColor = (position: number) => {
    if (position === 1) return 'bg-yellow-500';
    if (position <= 3) return 'bg-gray-400';
    if (position <= 5) return 'bg-blue-500';
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
              <h1 className="text-3xl font-bold tracking-tight">Constructors</h1>
              <p className="text-muted-foreground">Team statistics and championship standings</p>
            </div>
          </div>

          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search constructors..."
              className="w-full rounded-lg bg-background pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          {(constructorsLoading || standingsLoading) && (
            <Card>
              <CardContent className="flex items-center justify-center py-12">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </CardContent>
            </Card>
          )}

          {!constructorsLoading && !standingsLoading && (
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {filteredConstructors.map((constructor: Constructor) => {
                const standing = getConstructorStanding(constructor.id);
                return (
                  <Card 
                    key={constructor.id} 
                    className="cursor-pointer hover:shadow-lg transition-shadow"
                    onClick={() => setSelectedConstructorId(constructor.id)}
                  >
                    <CardHeader>
                      <div className="flex items-center gap-4">
                        <div className="h-16 w-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                          <Building2 className="h-8 w-8 text-white" />
                        </div>
                        <div className="flex-1">
                          <CardTitle className="text-lg">{constructor.name}</CardTitle>
                          <CardDescription>{constructor.nationality}</CardDescription>
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
                      </div>
                    </CardContent>
                  </Card>
                );
              })}
            </div>
          )}

          {!constructorsLoading && filteredConstructors.length === 0 && (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12 text-center">
                <Search className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-semibold mb-2">No constructors found</p>
                <p className="text-sm text-muted-foreground">
                  Try adjusting your search criteria
                </p>
              </CardContent>
            </Card>
          )}

          <Dialog open={!!selectedConstructorId} onOpenChange={(open) => !open && setSelectedConstructorId(null)}>
            <DialogContent className="max-w-2xl">
              <DialogHeader>
                <DialogTitle>
                  {selectedConstructor?.name}
                </DialogTitle>
              </DialogHeader>
              
              {selectedConstructor && (
                <div className="space-y-6">
                  <div className="grid gap-4 md:grid-cols-2">
                    <Card>
                      <CardHeader>
                        <CardTitle>Team Information</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Name</span>
                          <span className="font-semibold">{selectedConstructor.name}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Nationality</span>
                          <span className="font-semibold">{selectedConstructor.nationality}</span>
                        </div>
                        {selectedConstructor.url && (
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">Website</span>
                            <a 
                              href={selectedConstructor.url} 
                              target="_blank" 
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:underline"
                            >
                              Visit
                            </a>
                          </div>
                        )}
                      </CardContent>
                    </Card>

                    <Card>
                      <CardHeader>
                        <CardTitle>Season Statistics</CardTitle>
                      </CardHeader>
                      <CardContent className="space-y-2">
                        {getConstructorStanding(selectedConstructor.id) ? (
                          <>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Championship Position</span>
                              <span className="font-semibold">{getConstructorStanding(selectedConstructor.id)?.position}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Points</span>
                              <span className="font-semibold">{getConstructorStanding(selectedConstructor.id)?.points}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-muted-foreground">Wins</span>
                              <span className="font-semibold">{getConstructorStanding(selectedConstructor.id)?.wins}</span>
                            </div>
                          </>
                        ) : (
                          <p className="text-sm text-muted-foreground">No standings data available</p>
                        )}
                      </CardContent>
                    </Card>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>
        </main>
      </div>
    </div>
  );
}