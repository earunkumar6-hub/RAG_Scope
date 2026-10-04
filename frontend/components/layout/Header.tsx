"use client";

import { useQuery } from "@tanstack/react-query";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { cn } from "cn";

import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { api } from "@/lib/api";

function HealthDot() {
  const { data, isError } = useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: 30_000,
  });
  const color = isError ? "bg-red-500" : data?.status === "ok" ? "bg-emerald-500" : data ? "bg-amber-500" : "bg-muted-foreground";
  const label = isError ? "Backend unreachable" : data ? `Backend ${data.status}` : "Checking backend…";
  return (
    <Tooltip>
      <TooltipTrigger render={<span className="flex items-center gap-2 text-xs text-muted-foreground" />}>
        <span className={cn("size-2 rounded-full", color)} />
        {label}
      </TooltipTrigger>
      <TooltipContent>
        {data ? (
          <ul className="space-y-0.5">
            {Object.entries(data.components).map(([name, c]) => (
              <li key={name}>
                {name}: {c.status} {c.detail && `- ${c.detail}`}
              </li>
            ))}
          </ul>
        ) : (
          label
        )}
      </TooltipContent>
    </Tooltip>
  );
}

export function Header({ title }: { title: string }) {
  const { resolvedTheme, setTheme } = useTheme();
  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b px-4">
      <h1 className="text-sm font-semibold">{title}</h1>
      <div className="ml-auto flex items-center gap-3">
        <HealthDot />
        <Button
          variant="ghost"
          size="icon-sm"
          aria-label="Toggle theme"
          onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
        >
          <Sun className="hidden dark:block" />
          <Moon className="dark:hidden" />
        </Button>
      </div>
    </header>
  );
}
