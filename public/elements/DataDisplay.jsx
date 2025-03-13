import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Separator } from "@/components/ui/separator"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Button } from "@/components/ui/button"
import { Copy, ChevronDown, ChevronRight } from "lucide-react"
import { useState } from "react"

export default function DataDisplay() {
  // Data is passed via props
  const data = props.data || {};
  const title = props.title || "Data";
  const badge = props.badge || null;
  const maxHeight = props.maxHeight || "300px";
  const showScrollArea = props.showScrollArea !== false;
  const collapsible = props.collapsible !== false;
  const theme = props.theme || "default"; // default, compact, or expanded
  
  // State for collapsible sections
  const [expandedSections, setExpandedSections] = useState({});
  
  // Toggle section expansion
  const toggleSection = (key) => {
    setExpandedSections(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };
  
  // Copy value to clipboard
  const copyToClipboard = (value) => {
    let textToCopy;
    
    if (Array.isArray(value)) {
      // Format arrays properly for copying
      textToCopy = JSON.stringify(value);
    } else if (typeof value === 'object' && value !== null) {
      textToCopy = JSON.stringify(value, null, 2);
    } else {
      textToCopy = String(value);
    }
    
    navigator.clipboard.writeText(textToCopy);
    // Could use sonner for toast notification here
  };

  // Helper to format values with scientific notation for very small/large numbers
  const formatValue = (value) => {
    if (value === undefined || value === null) return "N/A";
    
    if (Array.isArray(value) || (typeof value === 'object' && value.length)) {
      return formatArray(value);
    }
    
    if (typeof value === 'object' && value !== null) {
      return null; // Handled separately in the render
    }
    
    if (typeof value === 'number') {
      // Use scientific notation for very small or large numbers
      if (Math.abs(value) < 0.001 || Math.abs(value) > 10000) {
        return value.toExponential(4);
      }
      // Format with appropriate precision
      return Number.isInteger(value) ? value.toString() : value.toFixed(4).replace(/\.?0+$/, '');
    }
    
    if (typeof value === 'boolean') {
      return value ? "true" : "false";
    }
    
    return String(value);
  };
  
  // Helper function to format arrays nicely
  const formatArray = (arr) => {
    if (!arr) return "N/A";
    if (typeof arr === 'string') return arr;
    
    try {
      // Handle both array-like objects and actual arrays
      if (arr.length > 10) {
        const displayed = Array.from(arr).slice(0, 10);
        return `[${displayed.map(val => 
          typeof val === 'object' ? '{...}' : 
          typeof val === 'number' ? formatValue(val) : 
          JSON.stringify(val)
        ).join(", ")}, ... ${arr.length - 10} more]`;
      }
      
      return `[${Array.from(arr).map(val => 
        typeof val === 'object' ? '{...}' : 
        typeof val === 'number' ? formatValue(val) : 
        JSON.stringify(val)
      ).join(", ")}]`;
    } catch (e) {
      return String(arr);
    }
  };
  
  // Helper function to render nested objects
  const renderNestedObject = (obj, path = "", level = 0) => {
    if (!obj || typeof obj !== 'object') return <span>{String(obj || "N/A")}</span>;
    
    if (Array.isArray(obj)) {
      return (
        <div className="flex items-center justify-between">
          <span className="text-sm font-mono">{formatArray(obj)}</span>
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-5 w-5 p-0 opacity-50 hover:opacity-100 ml-2" 
            onClick={() => copyToClipboard(obj)}
          >
            <Copy className="h-3 w-3" />
          </Button>
        </div>
      );
    }
    
    const isExpanded = expandedSections[path] !== false; // Default to expanded
    
    return (
      <div className={`space-y-1 ${level > 0 ? "pl-4" : ""}`}>
        {Object.entries(obj).map(([key, value], index) => {
          const currentPath = path ? `${path}.${key}` : key;
          const isObject = typeof value === 'object' && value !== null;
          
          return (
            <div key={currentPath} className="pt-1">
              {index > 0 && level === 0 && <Separator className="my-2" />}
              
              <div className="flex items-center">
                <div className="flex items-center gap-1 flex-grow">
                  {isObject && collapsible && (
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="h-5 w-5 p-0" 
                      onClick={() => toggleSection(currentPath)}
                    >
                      {expandedSections[currentPath] === false ? 
                        <ChevronRight className="h-4 w-4" /> : 
                        <ChevronDown className="h-4 w-4" />
                      }
                    </Button>
                  )}
                  <div className="font-medium text-sm">{key}</div>
                </div>
              </div>
              
              <div className={`
                text-sm 
                ${level > 0 ? "border-l-2 border-gray-200 dark:border-gray-700 pl-2" : ""}
                ${isObject && expandedSections[currentPath] === false ? "hidden" : ""}
              `}>
                {isObject ? 
                  renderNestedObject(value, currentPath, level + 1) : 
                  <div className="flex items-center justify-between">
                    <div className="font-mono">{formatValue(value)}</div>
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="h-5 w-5 p-0 opacity-50 hover:opacity-100 ml-2" 
                      onClick={() => copyToClipboard(value)}
                    >
                      <Copy className="h-3 w-3" />
                    </Button>
                  </div>
                }
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  // Apply theme styles
  const getThemeStyles = () => {
    switch (theme) {
      case 'compact':
        return "text-xs";
      case 'expanded':
        return "text-base";
      default:
        return "text-sm";
    }
  };

  return (
    <Card className={`w-full ${getThemeStyles()}`}>
      <CardHeader className="pb-2">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-medium">
            {title}
          </CardTitle>
          <div className="flex items-center gap-2">
            {badge && (
              <Badge variant="outline">{badge}</Badge>
            )}
            <Button 
              variant="ghost" 
              size="icon" 
              className="h-6 w-6 p-0" 
              onClick={() => copyToClipboard(data)}
              title="Copy all data"
            >
              <Copy className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {Object.keys(data).length === 0 ? (
          <div className="text-center text-muted-foreground py-4">No data available</div>
        ) : (
          showScrollArea ? (
            <ScrollArea style={{ height: maxHeight }} className="pr-4">
              {renderNestedObject(data)}
            </ScrollArea>
          ) : (
            renderNestedObject(data)
          )
        )}
      </CardContent>
    </Card>
  )
} 