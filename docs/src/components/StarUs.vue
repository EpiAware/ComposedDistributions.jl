<!-- MANAGED by EpiAwarePackageTools.scaffold — do not edit by hand. -->
<!-- GitHub stars widget for the docs navbar. Shows the repo's live star -->
<!-- count (from stargazers.data.ts) and links to the repository. -->
<!-- Adapted from https://github.com/LuxDL/DocumenterVitepress.jl -->
<script setup>
import {formatPrefix} from "d3-format";
import {data} from "./stargazers.data";

const props = defineProps({
  repoUrl: {
    type: String,
    default: "https://github.com/EpiAware/ComposedDistributions.jl"
  }
});

const formattedStarCount = data ? (data < 1000 ? `${data}` : formatPrefix(".1s", 1000)(data)) : "";
</script>

<template>
  <a
    target="_blank"
    data-decoration="&#x2605;"
    :title="data ? data.toLocaleString('en-US').concat(' GitHub stars') : ''"
    :href="repoUrl"
  >
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      width="20"
      height="20"
      fill="currentColor"
      style="vertical-align: middle; margin-right: 0.25rem; margin-left: 0.5rem;"
    >
      <path
        d="M12 .297C5.375.297 0 5.673 0 12.3c0 5.292 3.438 9.8 8.207 11.387.6.11.793-.26.793-.577 0-.285-.01-1.04-.015-2.04-3.338.727-4.042-1.61-4.042-1.61-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.807 1.305 3.493.997.107-.774.42-1.305.762-1.605-2.665-.3-5.467-1.333-5.467-5.931 0-1.31.47-2.382 1.236-3.222-.123-.303-.535-1.52.117-3.166 0 0 1.01-.323 3.31 1.23.96-.267 1.98-.4 3-.405 1.02.005 2.04.138 3 .405 2.3-1.553 3.31-1.23 3.31-1.23.653 1.646.24 2.863.117 3.166.765.84 1.236 1.912 1.236 3.222 0 4.61-2.807 5.625-5.477 5.921.43.372.823 1.102.823 2.222 0 1.606-.015 2.902-.015 3.293 0 .32.192.693.8.577C20.565 22.1 24 17.588 24 12.297 24 5.673 18.627.297 12 .297z"
      />
    </svg>
    <span>{{ formattedStarCount }}</span>
  </a>
</template>

<style scoped>
a {
  font-size: 14px;
  font-weight: 500;
  text-wrap: nowrap;
  display: flex;
  align-items: center;
  gap: 4px;
}

a[target="_blank"] {
  align-items: center;
  gap: 0.25rem;
  text-decoration: none;
}

a[target="_blank"]:hover span,
a[target="_blank"][data-decoration]::after {
  color: var(--vp-c-brand-1);
  transition: color 0.25s;
}

a[target="_blank"][data-decoration]::after {
  content: attr(data-decoration);
}

a[target="_blank"]:not(:hover, :focus)::after {
  color: inherit;
}
</style>
