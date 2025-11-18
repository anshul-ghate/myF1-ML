import { useDriverStandings, useConstructorStandings, useUpcomingRaces } from "@/hooks/useBackendData";

const DashboardPage = () => {
  const { data: driverStandings, isLoading: isLoadingDrivers, error: errorDrivers } = useDriverStandings(2024);
  const { data: constructorStandings, isLoading: isLoadingConstructors, error: errorConstructors } = useConstructorStandings(2024);
  const { data: upcomingRaces, isLoading: isLoadingRaces, error: errorRaces } = useUpcomingRaces();

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">F1 Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="border rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-2">Driver Standings</h2>
          {isLoadingDrivers && <p>Loading...</p>}
          {errorDrivers && <p className="text-red-500">{errorDrivers.message}</p>}
          {driverStandings && (
            <ul>
              {driverStandings.map((driver) => (
                <li key={driver.driver_id} className="flex justify-between">
                  <span>{driver.position}. {driver.driver_name}</span>
                  <span>{driver.points}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="border rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-2">Constructor Standings</h2>
          {isLoadingConstructors && <p>Loading...</p>}
          {errorConstructors && <p className="text-red-500">{errorConstructors.message}</p>}
          {constructorStandings && (
            <ul>
              {constructorStandings.map((constructor) => (
                <li key={constructor.constructor_id} className="flex justify-between">
                  <span>{constructor.position}. {constructor.constructor_name}</span>
                  <span>{constructor.points}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div className="border rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-2">Upcoming Races</h2>
          {isLoadingRaces && <p>Loading...</p>}
          {errorRaces && <p className="text-red-500">{errorRaces.message}</p>}
          {upcomingRaces && (
            <ul>
              {upcomingRaces.map((race) => (
                <li key={race.race_id}>{race.race_name} - {new Date(race.race_date).toLocaleDateString()}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;
