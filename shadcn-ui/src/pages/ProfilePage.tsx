import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

export default function ProfilePage() {
  return (
    <div className="flex-1 space-y-4 p-4 md:p-8 pt-6">
        <div className="flex items-center justify-center">
            <Card className="w-full max-w-2xl">
                <CardHeader>
                    <CardTitle className="text-2xl">User Profile</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col items-center gap-6">
                    <Avatar className="h-24 w-24">
                        <AvatarImage src="https://github.com/shadcn.png" alt="@shadcn" />
                        <AvatarFallback>MV</AvatarFallback>
                    </Avatar>
                    <div className="text-center">
                        <h2 className="text-xl font-semibold">Max Verstappen</h2>
                        <p className="text-muted-foreground">max@redbullracing.com</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4 w-full text-sm">
                        <div>
                            <p className="font-medium text-muted-foreground">Favorite Driver</p>
                            <p>Max Verstappen</p>
                        </div>
                        <div>
                            <p className="font-medium text-muted-foreground">Favorite Team</p>
                            <p>Red Bull Racing</p>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    </div>
  );
}