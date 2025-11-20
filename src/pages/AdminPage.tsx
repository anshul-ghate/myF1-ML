import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Loader2, Database, Brain, RefreshCw } from 'lucide-react';
import { triggerDataSync, triggerPredictionGeneration } from '@/lib/supabaseService';

export default function AdminPage() {
  const [syncing, setSyncing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  const handleDataSync = async () => {
    setSyncing(true);
    setMessage(null);
    try {
      await triggerDataSync();
      setMessage({ type: 'success', text: 'Data sync completed successfully! All F1 data has been updated.' });
    } catch (error) {
      setMessage({ type: 'error', text: `Data sync failed: ${error instanceof Error ? error.message : 'Unknown error'}` });
    } finally {
      setSyncing(false);
    }
  };

  const handleGeneratePredictions = async () => {
    setGenerating(true);
    setMessage(null);
    try {
      await triggerPredictionGeneration();
      setMessage({ type: 'success', text: 'Predictions generated successfully for the upcoming race!' });
    } catch (error) {
      setMessage({ type: 'error', text: `Prediction generation failed: ${error instanceof Error ? error.message : 'Unknown error'}` });
    } finally {
      setGenerating(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        <div>
          <h1 className="text-4xl font-bold mb-2">Backend Administration</h1>
          <p className="text-muted-foreground">Manage F1 data synchronization and prediction generation</p>
        </div>

        {message && (
          <Alert variant={message.type === 'error' ? 'destructive' : 'default'}>
            <AlertDescription>{message.text}</AlertDescription>
          </Alert>
        )}

        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Database className="h-5 w-5" />
                Data Synchronization
              </CardTitle>
              <CardDescription>Sync latest F1 data from Ergast API including drivers, constructors, circuits, races, and results</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={handleDataSync} disabled={syncing} className="w-full">
                {syncing ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Syncing Data...
                  </>
                ) : (
                  <>
                    <RefreshCw className="mr-2 h-4 w-4" />
                    Sync F1 Data
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Generate Predictions
              </CardTitle>
              <CardDescription>Generate ML-powered predictions for the upcoming race based on historical performance data</CardDescription>
            </CardHeader>
            <CardContent>
              <Button onClick={handleGeneratePredictions} disabled={generating} className="w-full">
                {generating ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Brain className="mr-2 h-4 w-4" />
                    Generate Predictions
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Setup Instructions</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">Initial Setup</h3>
              <ol className="list-decimal list-inside space-y-2 text-sm text-muted-foreground">
                <li>Click "Sync F1 Data" to populate the database with current season data</li>
                <li>Wait for the sync to complete (this may take 30-60 seconds)</li>
                <li>Click "Generate Predictions" to create predictions for the upcoming race</li>
                <li>Return to the dashboard to view the updated data</li>
              </ol>
            </div>
            <div>
              <h3 className="font-semibold mb-2">Regular Maintenance</h3>
              <ul className="list-disc list-inside space-y-2 text-sm text-muted-foreground">
                <li>Run data sync after each race to update results</li>
                <li>Generate new predictions before each race weekend</li>
                <li>Data sync can be run daily to catch any updates</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}