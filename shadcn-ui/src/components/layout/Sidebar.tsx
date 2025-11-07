import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Home, Trophy, Users, Building2, TrendingUp, Zap } from 'lucide-react';

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Predictions', href: '/predictions', icon: TrendingUp },
  { name: 'Strategy', href: '/strategy', icon: Zap },
  { name: 'Drivers', href: '/drivers', icon: Users },
  { name: 'Constructors', href: '/constructors', icon: Building2 },
];

export function Sidebar() {
  const location = useLocation();

  return (
    <div className="hidden border-r bg-muted/40 md:block">
      <div className="flex h-full max-h-screen flex-col gap-2">
        <div className="flex h-14 items-center border-b px-4 lg:h-[60px] lg:px-6">
          <Link to="/" className="flex items-center gap-2 font-semibold">
            <Trophy className="h-6 w-6 text-red-600" />
            <span className="text-lg">F1 Analytics</span>
          </Link>
        </div>
        <div className="flex-1">
          <nav className="grid items-start px-2 text-sm font-medium lg:px-4 gap-1">
            {navigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={cn(
                    'flex items-center gap-3 rounded-lg px-3 py-2 transition-all hover:text-primary',
                    isActive
                      ? 'bg-muted text-primary'
                      : 'text-muted-foreground hover:bg-muted/50'
                  )}
                >
                  <item.icon className="h-4 w-4" />
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>
      </div>
    </div>
  );
}