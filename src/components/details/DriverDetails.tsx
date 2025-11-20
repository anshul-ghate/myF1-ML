import { Driver, DriverStanding } from '@/data/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

interface DriverDetailsProps {
  driver: Driver;
  standing: DriverStanding;
}

export default function DriverDetails({ driver, standing }: DriverDetailsProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center gap-4">
        <Avatar className="h-16 w-16">
          <AvatarImage src="/assets/driver-placeholder_variant_2.png" alt={driver.givenName} />
          <AvatarFallback>{driver.code}</AvatarFallback>
        </Avatar>
        <div>
          <CardTitle className="text-2xl">{driver.givenName} {driver.familyName}</CardTitle>
          <p className="text-muted-foreground">#{driver.permanentNumber}</p>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="font-semibold">Nationality</p>
            <p>{driver.nationality}</p>
          </div>
          <div>
            <p className="font-semibold">Date of Birth</p>
            <p>{new Date(driver.dateOfBirth).toLocaleDateString()}</p>
          </div>
          <div>
            <p className="font-semibold">Season Wins</p>
            <p>{standing.wins}</p>
          </div>
          <div>
            <p className="font-semibold">Biography</p>
            <a href={driver.url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
              Wikipedia
            </a>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}