/**
 * Backend Status Indicator Component
 * Shows connection status to Python backend
 */

import React from 'react';
import { useBackendHealth } from '@/hooks/useBackendData';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { CheckCircle2, XCircle, Loader2 } from 'lucide-react';

export function BackendStatus() {
  const { isHealthy, checking } = useBackendHealth();

  if (checking) {
    return (
      <Alert className="mb-4">
        <Loader2 className="h-4 w-4 animate-spin" />
        <AlertDescription>
          Checking backend connection...
        </AlertDescription>
      </Alert>
    );
  }

  if (!isHealthy) {
    return (
      <Alert variant="destructive" className="mb-4">
        <XCircle className="h-4 w-4" />
        <AlertDescription>
          Backend is not responding. Make sure the Python backend is running at{' '}
          <code className="bg-black/10 px-1 py-0.5 rounded">
            {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
          </code>
        </AlertDescription>
      </Alert>
    );
  }

  return (
    <Alert className="mb-4 border-green-500 bg-green-50">
      <CheckCircle2 className="h-4 w-4 text-green-600" />
      <AlertDescription className="text-green-800">
        Connected to backend successfully
      </AlertDescription>
    </Alert>
  );
}