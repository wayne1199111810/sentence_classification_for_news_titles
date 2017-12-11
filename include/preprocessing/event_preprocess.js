const getCategoryWithTitle = infos => [infos[4], infos[1]].join(' ');

const fs = require('fs');
const filename = 'news';
const file = fs.readFileSync(filename, 'utf8');
const lines = file.split('\n');
// const categoriesWithTitle = lines.map(line => getCategoryWithTitle(line.split('\t')));
// fs.writeFileSync('corpora', categoriesWithTitle.join('\n'), 'utf8');
const categories = lines.map(line => {
    const c = line.split('\t')[4];
    if (c === undefined) {
        console.log(line);
    }
    return c;
});
const counts = categories.reduce((result, category) => ((result[category] += 1), result), { b: 0, m: 0, t: 0, e: 0 });
console.log(counts);
