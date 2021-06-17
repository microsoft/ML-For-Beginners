module.exports = {
	contents: ['docs/_sidebar.md'], // array of "table of contents" files path
	pathToPublic: 'pdf/readme.pdf', // path where pdf will stored
	pdfOptions: '<options for puppeteer.pdf()>', // reference: https://github.com/GoogleChrome/puppeteer/blob/master/docs/api.md#pagepdfoptions
	removeTemp: true, // remove generated .md and .html or not
	emulateMedia: 'screen', // mediaType, emulating by puppeteer for rendering pdf, 'print' by default (reference: https://github.com/GoogleChrome/puppeteer/blob/master/docs/api.md#pageemulatemediamediatype)
};
