"use client";

import { FileText, FlaskConical, Network, Search, Settings, Upload } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "cn";

const NAV = [
  { href: "/ingest", label: "Ingest", icon: Upload },
  { href: "/query", label: "Query", icon: Search },
  { href: "/documents", label: "Documents", icon: FileText },
  { href: "/graph", label: "Graph", icon: Network },
  { href: "/eval", label: "Eval", icon: FlaskConical },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <nav className="flex w-14 shrink-0 flex-col gap-1 border-r bg-sidebar p-2 lg:w-48">
      <div className="mb-3 hidden px-2 pt-1 text-sm font-semibold lg:block">RAGScope</div>
      {NAV.map(({ href, label, icon: Icon }) => (
        <Link
          key={href}
          href={href}
          title={label}
          aria-label={label}
          className={cn(
            "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm hover:bg-sidebar-accent",
            pathname.startsWith(href) && "bg-sidebar-accent font-medium",
          )}
        >
          <Icon className="size-4 shrink-0" />
          <span className="hidden lg:inline">{label}</span>
        </Link>
      ))}
    </nav>
  );
}
