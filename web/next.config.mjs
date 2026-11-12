export default {
  distDir: process.env.GM_NEXT_DIST || ".next",
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination:
          (process.env.GM_API_ORIGIN || "http://127.0.0.1:8100") +
          "/api/:path*",
      },
    ];
  },
};
