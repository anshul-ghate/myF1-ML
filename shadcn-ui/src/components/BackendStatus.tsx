/**
 * Backend Status Indicator
 * Shows connection status to Python FastAPI backend
 */
import { useBackendHealth } from '@/hooks/useBackendData';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react';

export function BackendStatus() {
  const { data, isLoading, isError } = useBackendHealth();

  if (isLoading) {
    return (
      <Alert className="mb-4">
        <Loader2 className="h-4 w-4 animate-spin" />
        <AlertDescription>Connecting to backend...</AlertDescription>
      </Alert>
    );
  }

  if (isError) {
    return (
      <Alert variant="destructive" className="mb-4">
        <XCircle className="h-4 w-4" />
        <AlertDescription>
          Backend offline. Please start the Python backend server at http://localhost:8000
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Alert className="mb-4 border-green-500 bg-green-50 dark:bg-green-950">
      <CheckCircle2 className="h-4 w-4 text-green-600" />
      <AlertDescription className="text-green-800 dark:text-green-200">
        Connected to backend v{data?.version}
      </AlertDescription>
    </Alert>
  );
}