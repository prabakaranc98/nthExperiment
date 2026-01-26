"use client";

import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import type { MetricData } from "@/types";

interface MetricInputProps {
  metric: MetricData;
  value: unknown;
  onChange: (value: unknown) => void;
}

export function MetricInput({ metric, value, onChange }: MetricInputProps) {
  const config = metric.config ? JSON.parse(metric.config) : {};

  switch (metric.type) {
    case "boolean":
      return (
        <div className="flex items-center space-x-3">
          <Switch
            checked={Boolean(value)}
            onCheckedChange={onChange}
            id={metric.id}
          />
          <Label htmlFor={metric.id} className="text-sm text-muted-foreground">
            {value ? "Yes" : "No"}
          </Label>
        </div>
      );

    case "number":
      return (
        <div className="flex items-center gap-2">
          <Input
            type="number"
            value={value !== undefined ? String(value) : ""}
            onChange={(e) => {
              const num = parseFloat(e.target.value);
              onChange(isNaN(num) ? undefined : num);
            }}
            min={config.min}
            max={config.max}
            className="max-w-32"
          />
          {config.unit && (
            <span className="text-sm text-muted-foreground">{config.unit}</span>
          )}
        </div>
      );

    case "scale":
      const min = config.min ?? 1;
      const max = config.max ?? 10;
      const currentValue = value !== undefined ? Number(value) : min;

      return (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">{min}</span>
            <span className="text-2xl font-bold">{currentValue}</span>
            <span className="text-sm text-muted-foreground">{max}</span>
          </div>
          <Slider
            value={[currentValue]}
            onValueChange={(values) => onChange(values[0])}
            min={min}
            max={max}
            step={1}
            className="w-full"
          />
          {config.labels && (
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>{config.labels[min] || ""}</span>
              <span>{config.labels[max] || ""}</span>
            </div>
          )}
        </div>
      );

    case "text":
      return (
        <Textarea
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          placeholder="Enter your response..."
          rows={3}
        />
      );

    case "duration":
      const unit = config.unit || "minutes";
      return (
        <div className="flex items-center gap-2">
          <Input
            type="number"
            value={value !== undefined ? String(value) : ""}
            onChange={(e) => {
              const num = parseFloat(e.target.value);
              onChange(isNaN(num) ? undefined : num);
            }}
            min={0}
            className="max-w-32"
          />
          <span className="text-sm text-muted-foreground">{unit}</span>
        </div>
      );

    case "select":
      const options = config.options || [];
      return (
        <Select
          value={String(value ?? "")}
          onValueChange={(v) => onChange(v)}
        >
          <SelectTrigger className="max-w-xs">
            <SelectValue placeholder="Select an option" />
          </SelectTrigger>
          <SelectContent>
            {options.map((option: string) => (
              <SelectItem key={option} value={option}>
                {option}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );

    case "multi-select":
      const multiOptions = config.options || [];
      const selectedValues = Array.isArray(value) ? value : [];

      return (
        <div className="space-y-2">
          {multiOptions.map((option: string) => (
            <div key={option} className="flex items-center space-x-2">
              <Checkbox
                id={`${metric.id}-${option}`}
                checked={selectedValues.includes(option)}
                onCheckedChange={(checked) => {
                  if (checked) {
                    onChange([...selectedValues, option]);
                  } else {
                    onChange(selectedValues.filter((v: string) => v !== option));
                  }
                }}
              />
              <Label
                htmlFor={`${metric.id}-${option}`}
                className="text-sm font-normal"
              >
                {option}
              </Label>
            </div>
          ))}
        </div>
      );

    default:
      return (
        <Input
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
        />
      );
  }
}
