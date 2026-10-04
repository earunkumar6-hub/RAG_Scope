"use client";

import { FileUp, X } from "lucide-react";
import { useRef, useState } from "react";
import { cn } from "cn";

import { Button } from "@/components/ui/button";

export const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".md", ".txt"];
export const MAX_FILES = 10;
export const MAX_FILE_BYTES = 25 * 1024 * 1024;

/** Client-side checks mirroring S1 (the backend re-checks, including magic bytes). */
export function fileProblems(files: File[]): string[] {
  const problems: string[] = [];
  if (files.length > MAX_FILES) problems.push(`At most ${MAX_FILES} files per run`);
  for (const f of files) {
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!ALLOWED_EXTENSIONS.includes(ext)) problems.push(`${f.name}: extension must be one of ${ALLOWED_EXTENSIONS.join(" ")}`);
    else if (f.size > MAX_FILE_BYTES) problems.push(`${f.name}: larger than 25 MB`);
  }
  return problems;
}

export function UploadZone({ files, onChange }: { files: File[]; onChange: (files: File[]) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const add = (list: FileList | null) => {
    if (!list) return;
    const byName = new Map(files.map((f) => [f.name, f]));
    Array.from(list).forEach((f) => byName.set(f.name, f));
    onChange([...byName.values()]);
  };
  return (
    <div className="space-y-2">
      <div
        role="button"
        tabIndex={0}
        onClick={() => input.current?.click()}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && input.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          add(e.dataTransfer.files);
        }}
        className={cn(
          "flex cursor-pointer flex-col items-center gap-1 rounded-lg border-2 border-dashed p-6 text-center text-sm text-muted-foreground transition-colors hover:border-ring",
          dragging && "border-primary bg-primary/5",
        )}
      >
        <FileUp className="size-6" />
        <span>Drop files here or click to choose</span>
        <span className="text-xs">{ALLOWED_EXTENSIONS.join(" ")} · up to {MAX_FILES} files · 25 MB each</span>
        <input
          ref={input}
          type="file"
          multiple
          accept={ALLOWED_EXTENSIONS.join(",")}
          className="hidden"
          data-testid="file-input"
          onChange={(e) => {
            add(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
      {files.length > 0 && (
        <ul className="flex flex-wrap gap-2">
          {files.map((f) => (
            <li key={f.name} className="flex items-center gap-1 rounded-md border px-2 py-0.5 text-xs">
              {f.name} <span className="text-muted-foreground">({Math.ceil(f.size / 1024)} KB)</span>
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label={`Remove ${f.name}`}
                onClick={() => onChange(files.filter((x) => x !== f))}
              >
                <X />
              </Button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
