import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import { ToastProvider } from "./context/ToastContext";
import { UploadProvider } from "./context/UploadContext";
import FightersPage from "./pages/FightersPage";
import JobsPage from "./pages/JobsPage";
import OverviewPage from "./pages/OverviewPage";
import SettingsPage from "./pages/SettingsPage";
import TrackingPage from "./pages/TrackingPage";
import UploadPage from "./pages/UploadPage";
import VideoDetailPage from "./pages/VideoDetailPage";
import VideoLibraryPage from "./pages/VideoLibraryPage";

export default function App() {
  return (
    <ToastProvider>
      <UploadProvider>
        <Layout>
          <Routes>
            <Route path="/" element={<OverviewPage />} />
            <Route path="/videos" element={<VideoLibraryPage />} />
            <Route path="/videos/:id" element={<VideoDetailPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/fighters" element={<FightersPage />} />
            <Route path="/fighters/:id" element={<FightersPage />} />
            <Route path="/tracking" element={<TrackingPage />} />
            <Route path="/tracking/:id" element={<TrackingPage />} />
            <Route path="/jobs" element={<JobsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Layout>
      </UploadProvider>
    </ToastProvider>
  );
}
