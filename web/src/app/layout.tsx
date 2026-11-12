import "./style.css";
export const metadata = {
  title: "Grounded Memory",
  description: "Observe. Ask. Remember.",
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
