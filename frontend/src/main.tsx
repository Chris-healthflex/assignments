import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import "./index.css";
import App from "./App.tsx";
import { ErrorBoundary } from "./components/ErrorBoundary.tsx";
import { CommandProvider } from "./components/CommandPalette.tsx";
import { ToastProvider } from "./components/ui/Toast.tsx";
import { UploadPage } from "./pages/UploadPage.tsx";
import { HistoryPage } from "./pages/HistoryPage.tsx";
import { AssessmentDetailPage } from "./pages/AssessmentDetailPage.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ErrorBoundary>
      <ToastProvider>
        <BrowserRouter>
          <CommandProvider>
            <Routes>
              <Route element={<App />}>
                <Route index element={<UploadPage />} />
                <Route path="history" element={<HistoryPage />} />
                <Route path="history/:id" element={<AssessmentDetailPage />} />
              </Route>
            </Routes>
          </CommandProvider>
        </BrowserRouter>
      </ToastProvider>
    </ErrorBoundary>
  </StrictMode>,
);
