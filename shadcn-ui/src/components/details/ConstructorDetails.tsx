import { Constructor, ConstructorStanding } from '@/data/types';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';

interface ConstructorDetailsProps {
  constructor: Constructor;
  standing: ConstructorStanding;
}

export default function ConstructorDetails({ constructor, standing }: ConstructorDetailsProps) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center gap-4">
        <Avatar className="h-16 w-16">
          <AvatarImage src="/assets/constructor-placeholder_variant_2.png" alt={constructor.name} />
          <AvatarFallback>{constructor.name.substring(0, 2)}</AvatarFallback>
        </Avatar>
        <div>
          <CardTitle className="text-2xl">{constructor.name}</CardTitle>
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="font-semibold">Nationality</p>
            <p>{constructor.nationality}</p>
          </div>
          <div>
            <p className="font-semibold">Season Wins</p>
            <p>{standing.wins}</p>
          </div>
          <div>
            <p className="font-semibold">History</p>
            <a href={constructor.url} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:underline">
              Wikipedia
            </a>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}