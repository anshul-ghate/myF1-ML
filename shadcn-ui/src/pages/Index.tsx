import { useState } from 'react';
import Header from '@/components/layout/Header';
import { Sidebar } from '@/components/layout/Sidebar';
import { RaceCountdown } from '@/components/dashboard/RaceCountdown';
import DriverStandings from '@/components/dashboard/DriverStandings';
import ConstructorStandings from '@/components/dashboard/ConstructorStandings';
import LastRaceResults from '@/components/dashboard/LastRaceResults';
import { Input } from '@/components/ui/input';
import { Search } from 'lucide-react';

export default function Index() {
  const [searchTerm, setSearchTerm] = useState('');

  return (
    <div className="grid min-h-screen w-full md:grid-cols-[220px_1fr] lg:grid-cols-[280px_1fr]">
      <Sidebar />
      <div className="flex flex-col">
        <Header />
        <main className="flex flex-1 flex-col gap-4 p-4 lg:gap-6 lg:p-6 bg-muted/40">
          <RaceCountdown />
          <LastRaceResults />
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              placeholder="Search standings..."
              className="w-full rounded-lg bg-background pl-8"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
            <div className="lg:col-span-4">
              <DriverStandings searchTerm={searchTerm} />
            </div>
            <div className="lg:col-span-3">
              <ConstructorStandings searchTerm={searchTerm} />
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}