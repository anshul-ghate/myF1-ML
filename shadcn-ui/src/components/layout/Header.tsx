import { Link } from "react-router-dom";
import { Menu, Package2, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet";
import { ModeToggle } from "../mode-toggle";

interface HeaderProps {
  onSearch: (query: string) => void;
}

export default function Header({ onSearch }: HeaderProps) {
  return (
    <header className="sticky top-0 flex h-16 items-center gap-4 border-b bg-background px-4 md:px-6 z-50">
      <nav className="hidden flex-col gap-6 text-lg font-medium md:flex md:flex-row md:items-center md:gap-5 md:text-sm lg:gap-6">
        <Link
          to="/"
          className="flex items-center gap-2 text-lg font-semibold md:text-base"
        >
          <Package2 className="h-6 w-6" />
          <span className="sr-only">F1 Analytics</span>
        </Link>
        <Link
          to="/"
          className="text-foreground transition-colors hover:text-foreground"
        >
          Dashboard
        </Link>
      </nav>
      <Sheet>
        <SheetTrigger asChild>
          <Button variant="outline" size="icon" className="shrink-0 md:hidden">
            <Menu className="h-5 w-5" />
            <span className="sr-only">Toggle navigation menu</span>
          </Button>
        </SheetTrigger>
        <SheetContent side="left">
          <nav className="grid gap-6 text-lg font-medium">
            <Link
              to="/"
              className="flex items-center gap-2 text-lg font-semibold"
            >
              <Package2 className="h-6 w-6" />
              <span className="sr-only">F1 Analytics</span>
            </Link>
            <Link to="/" className="hover:text-foreground">
              Dashboard
            </Link>
          </nav>
        </SheetContent>
      </Sheet>
      <div className="flex w-full items-center gap-4 md:ml-auto md:gap-2 lg:gap-4">
        <form className="ml-auto flex-1 sm:flex-initial" onSubmit={(e) => { e.preventDefault(); onSearch(e.currentTarget.search.value); }}>
          <div className="relative">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <Input
              type="search"
              name="search"
              placeholder="Search drivers or teams..."
              className="pl-8 sm:w-[300px] md:w-[200px] lg:w-[300px]"
            />
          </div>
        </form>
        <ModeToggle />
        {/* Reverted to simple buttons to fix a runtime error. The user dropdown menu will be re-added once the issue is resolved. */}
        <div className="flex items-center gap-2">
            <Button asChild variant="outline">
                <Link to="/login">Login</Link>
            </Button>
            <Button asChild>
                <Link to="/signup">Sign Up</Link>
            </Button>
        </div>
      </div>
    </header>
  );
}