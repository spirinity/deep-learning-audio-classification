"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { UPLOAD_LIMITS } from "@/lib/constants";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Loader2Icon, Music2Icon, UploadIcon } from "lucide-react";

interface Props {
  file: File | null;
  onSelect: (file: File) => void;
  onClear: () => void;
  onClassify: () => void;
  busy: boolean;
  disabled: boolean;
}

function validate(file: File): string | null {
  const name = file.name.toLowerCase();
  const okExt = UPLOAD_LIMITS.acceptExtensions.some((ext) => name.endsWith(ext));
  if (!okExt) return "Unsupported file type. Use .wav or .mp3.";
  const sizeMb = file.size / (1024 * 1024);
  if (sizeMb > UPLOAD_LIMITS.maxFileSizeMb) {
    return `File too large (${sizeMb.toFixed(1)} MB). Maximum is ${UPLOAD_LIMITS.maxFileSizeMb} MB.`;
  }
  return null;
}

export default function AudioUploader({
  file,
  onSelect,
  onClear,
  onClassify,
  busy,
  disabled,
}: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const previewUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : null),
    [file]
  );

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const f = files[0];
    const err = validate(f);
    if (err) {
      setLocalError(err);
      return;
    }
    setLocalError(null);
    onSelect(f);
  }

  if (!file) {
    return (
      <div>
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handleFiles(e.dataTransfer.files);
          }}
          className={`flex w-full flex-col items-center justify-center gap-3 rounded-xl border border-dashed px-6 py-12 text-center transition-colors ${
            dragging
              ? "border-primary bg-primary/5"
              : "border-border bg-card hover:border-muted-foreground/50 hover:bg-muted/40"
          }`}
        >
          <span className="flex size-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
            <UploadIcon className="size-6" />
          </span>
          <span className="text-sm font-medium">
            Drag &amp; drop audio here, or{" "}
            <span className="text-primary">browse</span>
          </span>
          <span className="text-xs text-muted-foreground">
            .wav or .mp3 · max {UPLOAD_LIMITS.maxFileSizeMb} MB · min 1 second
          </span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept={UPLOAD_LIMITS.acceptMime}
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        {localError && (
          <p className="mt-2 text-sm text-destructive">{localError}</p>
        )}
      </div>
    );
  }

  const sizeMb = file.size / (1024 * 1024);

  return (
    <Card>
      <CardContent>
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex min-w-0 items-center gap-2.5">
              <span className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary/15 text-primary">
                <Music2Icon className="size-4" />
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-medium">
                  {file.name}
                </span>
                <span className="block text-xs text-muted-foreground">
                  {sizeMb.toFixed(2)} MB
                </span>
              </span>
            </div>

            <div className="flex shrink-0 gap-2">
              <Button
                variant="outline"
                size="lg"
                onClick={() => {
                  setLocalError(null);
                  onClear();
                }}
                disabled={busy}
              >
                Clear
              </Button>
              <Button size="lg" onClick={onClassify} disabled={busy || disabled}>
                {busy ? (
                  <>
                    <Loader2Icon
                      data-icon="inline-start"
                      className="animate-spin"
                    />
                    Comparing…
                  </>
                ) : (
                  "Compare Methods"
                )}
              </Button>
            </div>
          </div>

          {previewUrl && (
            <audio
              controls
              src={previewUrl}
              className="w-full"
              preload="metadata"
            />
          )}
        </div>
        {disabled && !busy && (
          <p className="mt-2 text-xs text-muted-foreground">
            Backend model is not ready yet.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
