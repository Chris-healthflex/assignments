/**
 * Dotted paths, in exactly the form the server uses: `a.b[0].c`.
 *
 * The API keys every piece of evidence by one of these strings, so getting the
 * form fields to agree with them is what makes "this field was misheard" line
 * up with the input the clinician has to correct. Nothing here invents a path
 * format of its own.
 */

/** `objectiveAssessment.tests[1].left` -> `["objectiveAssessment","tests",1,"left"]` */
export function toSteps(path) {
  return path
    .replace(/\[(\d+)\]/g, ".$1")
    .split(".")
    .filter(Boolean)
    .map((step) => (/^\d+$/.test(step) ? Number(step) : step));
}

export function getAt(root, path) {
  return toSteps(path).reduce((node, step) => (node == null ? node : node[step]), root);
}

/**
 * Return a copy of `root` with `path` set to `value`.
 *
 * Immutable rather than in-place: React re-renders on identity, so mutating the
 * assessment would update the data and leave the screen showing the old value.
 * Untouched branches are shared, not cloned.
 */
export function setAt(root, path, value) {
  const steps = toSteps(path);
  const write = (node, index) => {
    if (index === steps.length) return value;
    const key = steps[index];
    if (Array.isArray(node)) {
      const copy = node.slice();
      copy[key] = write(node[key], index + 1);
      return copy;
    }
    return { ...node, [key]: write(node?.[key], index + 1) };
  };
  return write(root, 0);
}

/**
 * `["assessment","objectiveAssessment","tests",1,"left"]` -> the dotted path.
 *
 * The server prefixes `loc` with "assessment" so the pointer is valid against
 * the whole response body; the form is keyed by the contract-relative path, so
 * the prefix comes back off here.
 */
export function pathFromLoc(loc) {
  return loc
    .slice(1)
    .reduce(
      (acc, part) =>
        typeof part === "number" ? `${acc}[${part}]` : acc ? `${acc}.${part}` : String(part),
      "",
    );
}
