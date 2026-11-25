import React from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { AppLayout } from './layouts/AppLayout';
import { OverviewPage } from './pages/OverviewPage';
import { ImageAnalysisPage } from './pages/ImageAnalysisPage';
import { VideoAnalyticsPage } from './pages/VideoAnalyticsPage';
import { ImageProcessingLabPage } from './pages/ImageProcessingLabPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { DamageReportsPage } from './pages/DamageReportsPage';
import { DatasetExplorerPage } from './pages/DatasetExplorerPage';
import { SettingsPage } from './pages/SettingsPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          <Route index element={<OverviewPage />} />
          <Route path="image-analysis" element={<ImageAnalysisPage />} />
          <Route path="video-analytics" element={<VideoAnalyticsPage />} />
          <Route path="image-processing-lab" element={<ImageProcessingLabPage />} />
          <Route path="model-performance" element={<ModelPerformancePage />} />
          <Route path="damage-reports" element={<DamageReportsPage />} />
          <Route path="dataset-explorer" element={<DatasetExplorerPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};

export default App;
