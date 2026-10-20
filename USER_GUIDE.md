# CondensateLab — User Guide

> Chinese original: [docs/zh/使用说明.md](docs/zh/使用说明.md)

## Starting on Windows

Open the `Windows/CondensateLab` folder and double-click `CondensateLab.exe`. Keep the `_internal` folder in the same directory; when sharing the package, copy the entire `CondensateLab` folder.

The app runs in its own window and picks a free port automatically. No Python installation is needed; it uses the Windows Edge WebView2 component.

Task history is stored in the local `CondensateLab` data directory of the current user. Closing the window ends the current computation; availability of remote predictions depends on the corresponding services.

## Web sharing

The temporary web share URL is listed in a separate file (《分享地址.md》/ share-URL note) distributed next to this guide. Temporary sharing requires your machine and the sharing process to stay online.

For long-term deployment, use the `Dockerfile` in `source/`: create a Docker Space on Hugging Face and upload the contents of `source/`. Each of the two apps is deployed to its own Space. Cloud and local task histories are kept separately.

## Languages

The interface is available in Chinese, English, French, Spanish, Portuguese, and Arabic. Arabic uses a right-to-left layout. Raw results, errors, and some method hints returned by upstream services are kept in their original language.

## Current verification

The Windows package has passed local service startup and front-page checks. Visual inspection of the desktop window is not complete yet.
