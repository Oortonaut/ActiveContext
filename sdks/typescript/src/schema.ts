/**
 * Schema generation helpers.
 *
 * Provides builder functions for constructing NodeTypeSchema objects
 * declaratively. These schemas are advertised during the initialize
 * handshake and enable code generation, validation, and LLM documentation.
 *
 * @module schema
 */

import type {
  NodeTypeSchema,
  ConstructorSchema,
  ParamSchema,
  PropertySchema,
  MethodSchema,
} from "./types";

// ---------------------------------------------------------------------------
// Param builder
// ---------------------------------------------------------------------------

/**
 * Create a parameter schema.
 *
 * @param name - Parameter name.
 * @param options - Optional type, default, and description.
 * @returns A ParamSchema object.
 *
 * @example
 * ```ts
 * param("timeout", { type: "int", default: 120, description: "Timeout in seconds." })
 * ```
 */
export function param(
  name: string,
  options?: {
    type?: string;
    default?: unknown;
    description?: string;
  }
): ParamSchema {
  const schema: ParamSchema = { name };
  if (options?.type !== undefined) schema.type = options.type;
  if (options?.default !== undefined) schema.default = options.default;
  if (options?.description !== undefined) schema.description = options.description;
  return schema;
}

/**
 * Create a required parameter (no default value).
 *
 * @param name - Parameter name.
 * @param type - Type hint string.
 * @param description - Human-readable description.
 * @returns A ParamSchema with no default field.
 */
export function required(
  name: string,
  type = "str",
  description = ""
): ParamSchema {
  const schema: ParamSchema = { name, type };
  if (description) schema.description = description;
  return schema;
}

/**
 * Create an optional parameter with a default value.
 *
 * @param name - Parameter name.
 * @param defaultValue - Default value.
 * @param type - Type hint string.
 * @param description - Human-readable description.
 * @returns A ParamSchema with a default field.
 */
export function optional(
  name: string,
  defaultValue: unknown,
  type = "str",
  description = ""
): ParamSchema {
  const schema: ParamSchema = { name, type, default: defaultValue };
  if (description) schema.description = description;
  return schema;
}

// ---------------------------------------------------------------------------
// Property builder
// ---------------------------------------------------------------------------

/**
 * Create a property schema.
 *
 * @param name - Property name.
 * @param options - Optional type, readable, writable, and description.
 * @returns A PropertySchema object.
 *
 * @example
 * ```ts
 * property("is_complete", { type: "bool", readable: true, description: "Whether done." })
 * ```
 */
export function property(
  name: string,
  options?: {
    type?: string;
    readable?: boolean;
    writable?: boolean;
    description?: string;
  }
): PropertySchema {
  return {
    name,
    type: options?.type ?? "Any",
    readable: options?.readable ?? true,
    writable: options?.writable ?? false,
    description: options?.description ?? "",
  };
}

// ---------------------------------------------------------------------------
// Method builder
// ---------------------------------------------------------------------------

/**
 * Create a method schema.
 *
 * @param name - Method name.
 * @param options - Optional params, returns, description, and chainable flag.
 * @returns A MethodSchema object.
 *
 * @example
 * ```ts
 * method("cancel", { description: "Cancel the running command." })
 * method("SetTokens", {
 *   params: [param("tokens", { type: "int" })],
 *   returns: "self",
 *   chainable: true,
 * })
 * ```
 */
export function method(
  name: string,
  options?: {
    params?: ParamSchema[];
    returns?: string;
    description?: string;
    chainable?: boolean;
  }
): MethodSchema {
  return {
    name,
    params: options?.params ?? [],
    returns: options?.returns ?? "None",
    description: options?.description ?? "",
    chainable: options?.chainable ?? false,
  };
}

// ---------------------------------------------------------------------------
// Constructor builder
// ---------------------------------------------------------------------------

/**
 * Create a constructor schema.
 *
 * @param options - Positional, variadic, and named parameters.
 * @returns A ConstructorSchema object.
 *
 * @example
 * ```ts
 * constructor_({
 *   positional: [required("command", "str", "The command to execute.")],
 *   variadic: param("args", { type: "str", description: "Additional arguments." }),
 *   named: [optional("timeout", 120, "int", "Timeout in seconds.")],
 * })
 * ```
 */
export function constructor_(options?: {
  positional?: ParamSchema[];
  variadic?: ParamSchema | null;
  named?: ParamSchema[];
}): ConstructorSchema {
  return {
    positional: options?.positional ?? [],
    variadic: options?.variadic ?? null,
    named: options?.named ?? [],
  };
}

// ---------------------------------------------------------------------------
// Node type schema builder
// ---------------------------------------------------------------------------

/**
 * Create a complete node type schema.
 *
 * @param nodeType - Type identifier (e.g., "shell", "echo").
 * @param options - Description, constructor, properties, and methods.
 * @returns A NodeTypeSchema object ready for the initialize handshake.
 *
 * @example
 * ```ts
 * const echoSchema = nodeTypeSchema("echo", {
 *   description: "Echoes input text back.",
 *   constructor: constructor_({
 *     positional: [required("text", "str", "Text to echo.")],
 *   }),
 *   properties: [
 *     property("text", { type: "str", readable: true, description: "The echoed text." }),
 *   ],
 *   methods: [
 *     method("set_text", {
 *       params: [required("text", "str", "New text.")],
 *       description: "Update the echoed text.",
 *     }),
 *   ],
 * });
 * ```
 */
export function nodeTypeSchema(
  nodeType: string,
  options?: {
    description?: string;
    constructor?: ConstructorSchema;
    properties?: PropertySchema[];
    methods?: MethodSchema[];
  }
): NodeTypeSchema {
  return {
    node_type: nodeType,
    description: options?.description ?? "",
    constructor: options?.constructor ?? { positional: [], variadic: null, named: [] },
    properties: options?.properties ?? [],
    methods: options?.methods ?? [],
  };
}
