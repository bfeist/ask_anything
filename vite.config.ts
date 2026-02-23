import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import fs from "fs";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Local-dev plugin: serve video files from the external download directory
// and search-index data files from data/search_index/.
// ---------------------------------------------------------------------------
const VIDEO_DIR = "D:/ask_anything_ia_videos_raw";
const DATA_DIR = path.resolve(__dirname, "data");

function localMediaPlugin(): Plugin {
  return {
    name: "local-media-server",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url) return next();

        const decoded = decodeURIComponent(req.url);

        // Serve videos at /static_assets/videos/<filename>.mp4
        if (decoded.startsWith("/static_assets/videos/")) {
          const filename = decoded.slice("/static_assets/videos/".length);
          const filePath = path.join(VIDEO_DIR, filename);

          if (!fs.existsSync(filePath)) {
            res.statusCode = 404;
            res.end("Video not found");
            return;
          }

          const stat = fs.statSync(filePath);
          const range = req.headers.range;

          if (range) {
            // Support range requests for video seeking
            const parts = range.replace(/bytes=/, "").split("-");
            const start = parseInt(parts[0], 10);
            const end = parts[1] ? parseInt(parts[1], 10) : stat.size - 1;
            const chunkSize = end - start + 1;

            res.writeHead(206, {
              "Content-Range": `bytes ${start}-${end}/${stat.size}`,
              "Accept-Ranges": "bytes",
              "Content-Length": chunkSize,
              "Content-Type": "video/mp4",
            });
            fs.createReadStream(filePath, { start, end }).pipe(res);
          } else {
            res.writeHead(200, {
              "Content-Length": stat.size,
              "Content-Type": "video/mp4",
              "Accept-Ranges": "bytes",
            });
            fs.createReadStream(filePath).pipe(res);
          }
          return;
        }

        // Serve data files at /static_assets/data/<file>
        if (decoded.startsWith("/static_assets/data/")) {
          const relPath = decoded.slice("/static_assets/data/".length);
          const filePath = path.join(DATA_DIR, relPath);

          if (!fs.existsSync(filePath)) {
            res.statusCode = 404;
            res.end("Data file not found");
            return;
          }

          const stat = fs.statSync(filePath);
          const ext = path.extname(filePath);
          const contentType =
            ext === ".json"
              ? "application/json"
              : ext === ".bin"
                ? "application/octet-stream"
                : "application/octet-stream";

          res.writeHead(200, {
            "Content-Length": stat.size,
            "Content-Type": contentType,
            "Cache-Control": "no-cache",
          });
          fs.createReadStream(filePath).pipe(res);
          return;
        }

        next();
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), localMediaPlugin()],
  server: {
    host: "0.0.0.0",
    port: 9300,
    headers: {
      "Cross-Origin-Embedder-Policy": "require-corp",
      "Cross-Origin-Opener-Policy": "same-origin",
    },
  },
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
      components: path.resolve(__dirname, "./src/components"),
      assets: path.resolve(__dirname, "./src/assets"),
      styles: path.resolve(__dirname, "./src/styles"),
    },
  },
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
  worker: {
    format: "es",
  },
});
