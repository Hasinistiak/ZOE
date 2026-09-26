import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "fs";

export default defineConfig({
    plugins: [react()],

    server: {
        host: "0.0.0.0",
        port: 5173,

        https: {
            key: fs.readFileSync("./zoe.local+2-key.pem"),
            cert: fs.readFileSync("./zoe.local+2.pem"),
        },

        hmr: {
            host: "zoe.local",
            protocol: "wss",
            port: 5173,
        },

        proxy: {
            "/api": {
                target: "http://127.0.0.1:8000",
                changeOrigin: true,
            },
        },
    },
});